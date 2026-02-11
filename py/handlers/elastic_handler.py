import copy
import logging
import time
from enum import Enum
from typing import List

import elasticsearch
import isodate
import ndjson
import pandas as pd
import pytz
import requests

import config
from py.common.config_parser import parse_app_properties
from py.common.time_functions import convert_datetime_to_string_utc

logger = logging.getLogger(__name__)

try:
    from brcc_apis.elk_api import Elastic
except ModuleNotFoundError:
    from elasticsearch import Elasticsearch

    class Elastic:

        """
        Elastic client if no module is found

        :param server: elastic address
        :param debug: whether to be in debug mode
        """

        def __init__(self, server: str, debug: bool = False):
            """
            Constructor


            """
            self.server = server
            self.debug = debug
            self.client = Elasticsearch(self.server)


from datetime import datetime, timedelta

INITIAL_SCROLL_TIME = "5m"
CONSECUTIVE_SCROLL_TIME = "2m"
FIELD_NAME = "hits"
DOCUMENT_COUNT = 10000
DEFAULT_COLUMNS = ["value"]
SCROLL_ID_FIELD = '_scroll_id'
RESULT_FIELD = 'hits'
DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
MAGIC_KEYWORD = '_source'
COUNT_DOCUMENTS = 'count'
MAPPINGS_KEYWORD = 'mappings'
PROPERTIES_KEYWORD = 'properties'

ELASTIC_KEYWORDS = ["bool", "must", "should", "filter", "must_not",
                    "intervals", "match", "term", "terms", "terms_set",
                    "match_phrase", "match_phrase_prefix", "exists"
                    "multi_match", "combined_fields", "query_string", "range", "wildcard"]

NESTED_DICT_FIELD_DELIMITER = '.'

parse_app_properties(globals(), config.paths.config.elastic)
PY_ELASTICSEARCH_HOST = ELASTICSEARCH_HOST

PY_PROCUREMENT_PROPOSED_INDEX = PROCUREMENT_PROPOSED_INDEX
PY_PROCUREMENT_REALISED_INDEX = PROCUREMENT_REALISED_INDEX
PY_PROCUREMENT_ATC_INDEX = PROCUREMENT_ATC_INDEX
PY_PROCUREMENT_NCPB_INDEX = PROCUREMENT_NCPB_INDEX
PY_SIZING_CURRENT_BALANCING_STATE_INDEX = SIZING_CURRENT_BALANCING_STATE_INDEX ##[CHECK]
PY_PROCUREMENT_PROD_ATC_INDEX = PROCUREMENT_PROD_ATC_INDEX
PY_SIZING_AND_PROCUREMENT_LOGS_INDEX = SIZING_AND_PROCUREMENT_LOGS_INDEX
PY_AREA_INDEX = AREA_INDEX
PY_PROCUREMENT_EXCEEDED_INDEX = PROCUREMENT_EXCEEDED_INDEX

logger = logging.getLogger(__name__)


def parse_dict_flat_by_field_list(dict_to_flat: dict, field_list: list, delimiter = NESTED_DICT_FIELD_DELIMITER):
    """
    For elastic: parses nested dict to flat to load into dataframe

    :param dict_to_flat: input dictionary to flat
    :param field_list: field list to extract
    :param delimiter: Delimiter for multilevel index
    :return: dictionary
    """
    output = {}
    for field_value in field_list:
        if isinstance(field_value, str):
            field_value = [field_value]
        new_value = dict_to_flat
        for single_field in field_value:
            if not isinstance(new_value, dict):
                break
            new_value = new_value.get(single_field, {})
        if new_value is not None and new_value != {}:
            output[delimiter.join(field_value)] = new_value
    return output


def nested_dict_to_flat(input_dict: dict,
                        output_dict: dict = None,
                        keys: list = None,
                        delimiter: str = NESTED_DICT_FIELD_DELIMITER):
    """
    Flattens nested dict to single level one. Keys are converted to strings and joined together with delimiter
    For iterable values (list) counter number is added to the key

    :param input_dict: input dictionary to parse
    :param output_dict: output dictionary where values are gathered
    :param keys: running list of keys
    :param delimiter: character to be used to join keys together
    :return: updated output_dict
    """
    if output_dict is None:
        output_dict = {}
    if keys is None:
        keys = []
    if not isinstance(input_dict, dict):
        final_key = NESTED_DICT_FIELD_DELIMITER.join(keys)
        output_dict[final_key] = input_dict
        return output_dict
    for key, value in input_dict.items():
        new_keys = keys + [str(key)]
        if isinstance(value, dict):
            output_dict = nested_dict_to_flat(input_dict=value,
                                              output_dict=output_dict,
                                              keys=new_keys,
                                              delimiter=delimiter)
        elif isinstance(value, list):
            counter = 1
            for single_item in value:
                list_keys = new_keys + [str(counter)]
                output_dict = nested_dict_to_flat(input_dict=single_item,
                                                  output_dict=output_dict,
                                                  keys=list_keys,
                                                  delimiter=delimiter)
                counter = counter + 1
        else:
            final_key = NESTED_DICT_FIELD_DELIMITER.join(new_keys)
            output_dict[final_key] = value
    return output_dict


def factory(data, escape_attribute_prefix: str = '_'):
    """
    Factory method for generating dictionaries from objects

    :param data: data to parse
    :param escape_attribute_prefix: for escaping protected attributes
    :return:
    """
    if escape_attribute_prefix:
        output = dict(x for x in data if x[1] is not None and not str(x[0]).startswith(escape_attribute_prefix))
    else:
        output = dict(x for x in data if x[1] is not None)
    # escape enum
    output = {k: v.value if isinstance(v, Enum) else v for k, v in output.items()}
    # escape datetime
    output = {k: convert_datetime_to_string_utc(v) if isinstance(v, datetime) else v for k, v in output.items()}
    # escape duration
    output = {k: isodate.duration_isoformat(v) if isinstance(v, timedelta) else v for k, v in output.items()}
    return output


def send_dictionaries_to_elastic(input_list: list | dict,
                                 elastic_index_name: str,
                                 elastic_server: str):
    """
    Sends given input to elastic

    :param input_list: dictionary (id: content) or list (ids will be generated automatically)
    :param elastic_index_name: index where to store data
    :param elastic_server: address of server
    :return: True if all was sent successfully, false otherwise
    """
    response = ElkHandler.send_to_elastic_bulk(index=elastic_index_name,
                                               json_message_list=input_list,
                                               server=elastic_server)
    if response:
        pass
        # logger.info(f"{len(input_list)} values sent to {elastic_index_name}")
    else:
        logger.warning(f"Unable to send data to {elastic_index_name}")
    return response


class ElkHandler(Elastic):
    """
    For handling get and post to ELK for calculation of the reserves

    :param server: address of elk server
    :param initial_scroll_time: for scrolling initial value
    :param consecutive_scroll_time: for each consecutive scroll
    :param document_count: number of rows to be extracted
    :param field_name: where results are located
    :param date_time_format: date time format used by elasticsearch
    :param debug: if needed
    """

    def __init__(self,
                 server,
                 initial_scroll_time: str = INITIAL_SCROLL_TIME,
                 consecutive_scroll_time: str = CONSECUTIVE_SCROLL_TIME,
                 document_count: str = DOCUMENT_COUNT,
                 field_name: str = FIELD_NAME,
                 date_time_format: str = DATE_TIME_FORMAT,
                 debug: bool = False):
        """
        Constructor
        """
        self.initial_scroll_time = initial_scroll_time
        self.consecutive_scroll_time = consecutive_scroll_time
        self.document_count = document_count
        self.field_name = field_name
        self.date_time_format = date_time_format
        super().__init__(server=server, debug=debug)

    def get_document_count(self, index: str, query: dict = None):
        """
        Counts number of documents found in the index (by using the query)

        :param index: table where search from
        :param query: optional the query by which to search from
        :return: number of documents
        """
        if query is None:
            results = self.client.count(index=index)
        else:
            results = self.client.count(index=index, query=query)
        return results[COUNT_DOCUMENTS]


    def get_index_fields_from_response(self,
                                       index_mapping: dict,
                                       output_list: list = None,
                                       parent_key: list = None):
        """
        Gets list fo fields from input data recursively

        :param index_mapping: input from elastic
        :param output_list: list of keys
        :param parent_key: parent key name
        :return:
        """
        if not output_list:
            output_list = []
        if properties_dict := index_mapping.get(PROPERTIES_KEYWORD):
            for key, value in properties_dict.items():
                value = dict(value)
                if parent_key:
                    if isinstance(parent_key, str):
                        parent_key = [parent_key]
                    new_key = copy.deepcopy(parent_key)
                    new_key.append(key)
                else:
                    new_key = key
                if PROPERTIES_KEYWORD in value:
                    output_list = self.get_index_fields_from_response(index_mapping=value,
                                                                      output_list=output_list,
                                                                      parent_key=new_key)
                else:
                    output_list.append(new_key)
        return output_list

    def get_index_fields(self, index: str):
        """
        Returns list of fields

        :param index: table name
        :return: list of fields(columns) in the index
        """
        response = self.client.indices.get_mapping(index=index)
        first_index = next(iter(response.values()))

        if mappings := dict(first_index.get(MAPPINGS_KEYWORD, {})):
            output_list = self.get_index_fields_from_response(index_mapping=mappings)
            return output_list
        return None

    def get_data_by_scrolling(self,
                              query: dict,
                              index: str,
                              fields: [] = None):
        """
        Asks data from elk by scrolling
        Rework this: allocate memory to data structure beforehand and then start writing into it

        :param query: dictionary for asking
        :param index: index (table) from where to ask
        :param fields: fields (columns) to ask from index (table)
        :return:
        """
        if fields is None:
            result = self.client.search(index=index,
                                        query=query,
                                        size=self.document_count,
                                        scroll=self.initial_scroll_time)
        else:
            filtered_fields = []
            for field in filtered_fields:
                if isinstance(field, list):
                    field = NESTED_DICT_FIELD_DELIMITER.join(field)
                filtered_fields.append(field)
            result = self.client.search(index=index,
                                        query=query,
                                        source=filtered_fields,
                                        size=self.document_count,
                                        scroll=self.initial_scroll_time)

        scroll_id = result[SCROLL_ID_FIELD]
        # Extract and return the relevant data from the initial response
        hits = result[RESULT_FIELD][RESULT_FIELD]
        yield hits

        # Continue scrolling through the results until there are no more
        while hits:
            result = self.client.scroll(scroll_id=scroll_id, scroll=self.consecutive_scroll_time)
            hits = result[RESULT_FIELD][RESULT_FIELD]

            yield hits
        # Clear the scroll context after processing all results
        self.client.clear_scroll(scroll_id=scroll_id)

    def get_data(self,
                 query: dict,
                 index: str,
                 fields: list = None,
                 dict_to_flat: bool = True,
                 use_default_fields: bool = False):
        """
        Asks data from elk and stores it to numpy array
        For composing the query use https://test-rcc-logs.elering.sise/app/dev_tools#/console

        :param dict_to_flat:
        :param query: dictionary to query from elk
        :param index: index (table) from where to query
        :param fields: fields (columns) to be extracted, note that these must be strings
        :param use_default_fields: whether to use default fields and unpack them
        :return: pandas dataframe with fields as columns
        """
        # Get number of documents
        row_count = self.get_document_count(index=index, query=query)
        # Get columns if not specified
        if fields is None and use_default_fields:
            fields = [[field] if isinstance(field, str) else field for field in self.get_index_fields(index=index)]
        logger.info(f"Reading {index}:{row_count} documents found")
        counter = 1
        timer_start = time.time()
        # Gather all the results to list (of dictionaries)
        list_of_lines = []
        for hits in self.get_data_by_scrolling(query, index, fields):
            counter += 1
            for hit in hits:
                # Get content
                content = hit.get(MAGIC_KEYWORD, {})
                # if specific fields are requested get content by fields
                if fields:
                    content = parse_dict_flat_by_field_list(content, field_list=fields)
                # if no fields are presented but dict_to_flat flag is up flatten the dict
                elif dict_to_flat:
                    content = nested_dict_to_flat(input_dict=content)
                # if dict_to_flat is down and no fields are given then stuff the (nested) dict to dataframe
                if content is not None and content != {}:
                    list_of_lines.append(content)
        data_frame = pd.DataFrame(list_of_lines)
        timer_stop = time.time()
        logger.info(f"Reading {index}: done with {(timer_stop - timer_start):.2f}s for request")
        return data_frame

    @staticmethod
    def send_to_elastic_bulk(index: str,
                             json_message_list: List[dict] | dict,
                             id_from_metadata: bool = False,
                             id_metadata_list: List[str] | None = None,
                             server: str = "http://test-rcc-logs-master.elering.sise:9200",
                             iso_timestamp: str | None = None,
                             batch_size: int = 1000,
                             index_rollover: bool = True):
        """
        Method to send bulk message to ELK, allow pass custom ids to elastic

        :param index: index pattern in ELK
        :param json_message_list: list of messages in json format
        :param id_from_metadata:
        :param id_metadata_list:
        :param server: url of ELK server
        :param iso_timestamp: timestamp to be included in documents
        :param batch_size: maximum size of batch
        :param index_rollover: modifies given index by month indication
        :return:
        """
        try:
            if isinstance(json_message_list, List):
                return super().send_to_elastic_bulk(index=index,
                                                    json_message_list=json_message_list,
                                                    id_from_metadata=id_from_metadata,
                                                    id_metadata_list=id_metadata_list,
                                                    server=server,
                                                    iso_timestamp=iso_timestamp,
                                                    batch_size =batch_size,
                                                    index_rollover=index_rollover)
        except RuntimeError:
            if id_from_metadata:
                id_separator = "_"
                json_message_list = [value for element in json_message_list
                                     for value in ({"index": {"_index": index,
                                                              "_id": id_separator.join([str(element.get(key, ''))
                                                                                        for key in id_metadata_list])}},
                                                   element)]
            else:
                json_message_list = [value for element in json_message_list
                                     for value in ({"index": {"_index": index}}, element)]

        # Creating timestamp value if it is not provided in function call
        if not iso_timestamp:
            iso_timestamp = datetime.now(pytz.UTC).isoformat(sep="T")

        # Define server url with relevant index pattern
        if index_rollover:
            index = f"{index}-{datetime.today():%Y%m}"
        url = f"{server}/{index}/_bulk"

        if isinstance(json_message_list, dict):
            json_message_list = [value for key in json_message_list.keys()
                                 for value in ({"index": {"_index": index, "_id": key}},
                                               {**(json_message_list.get(key, {})), '@timestamp': iso_timestamp})]

        response_list = []
        for batch in range(0, len(json_message_list), batch_size):
            # Executing POST to push messages into ELK
            logger.debug(f"Sending batch ({batch}-{batch + batch_size})/{len(json_message_list)} to {url}")
            response = requests.post(url=url,
                                     data=(ndjson.dumps(json_message_list[batch:batch + batch_size]) + "\n").encode(),
                                     timeout=None,
                                     headers={"Content-Type": "application/x-ndjson"})
            logger.debug(f"ELK response: {response.content}")
            response_list.append(response.ok)

        return all(response_list)


def get_data_from_elastic(elk_server: str = PY_ELASTICSEARCH_HOST,
                          elastic_index: str = None,
                          elastic_query: dict = None,
                          dict_to_flat: bool = True,
                          use_default_fields: bool = True):
    """
    Gets data from elastic

    :param elk_server: address to the server
    :param elastic_index: index where data is stored
    :param elastic_query:
    :param use_default_fields:
    :param dict_to_flat:
    :return: dataframe with results
    """
    logger.info(f"Query: {elastic_query}")
    elk = ElkHandler(server=elk_server)
    response = elk.get_data(query=elastic_query,
                            index=str(elastic_index).removesuffix('*') + '*',
                            dict_to_flat=dict_to_flat,
                            use_default_fields=use_default_fields)
    logger.info(f"Received {len(response.index)} responses")
    return response


def get_data_from_elastic_by_time(start_time_value: str | datetime = None,
                                  end_time_value: str | datetime = None,
                                  elk_server: str = PY_ELASTICSEARCH_HOST,
                                  elastic_index: str = None,
                                  elastic_query: dict = None,
                                  time_interval_key: str | list = 'start_time',
                                  dict_to_flat: bool = True,
                                  use_default_fields: bool = False):
    """
    Gets data from elastic by time interval

    :param start_time_value: from where to start to query
    :param end_time_value: to where to query
    :param elk_server: address to the server
    :param elastic_index: index where data is stored
    :param time_interval_key: key by which to query
    :param elastic_query:
    :param use_default_fields:
    :param dict_to_flat:
    :return: dataframe with results
    """
    time_interval_key = [time_interval_key] if isinstance(time_interval_key, str) else time_interval_key
    start_time_dict = {}
    end_time_dict = {}
    if start_time_value is not None:
        start_time_dict = {"gte": convert_datetime_to_string_utc(start_time_value)}
    if end_time_value is not None:
        end_time_dict = {"lt" if len(time_interval_key) == 1 else "lte": convert_datetime_to_string_utc(end_time_value)}
    time_query = []
    if start_time_dict != {} or end_time_dict != {}:
        if len(time_interval_key) == 1:
            time_query = [{"range": {time_interval_key[0]: {**start_time_dict, **end_time_dict}}}]
        elif len(time_interval_key) >= 2:
            if start_time_dict != {}:
                time_query.append({"range": {time_interval_key[0]: start_time_dict}})
            if end_time_dict != {}:
                time_query.append({"range": {time_interval_key[1]: end_time_dict}})
    if len(time_query) >= 1:
        if not elastic_query:
            elastic_query = {"bool": {"must":  time_query}}
        else:
            logger.info(f"Adding additional parameters")
            elastic_query = merge_queries(elastic_query, time_query)
    if elastic_query is None:
        elastic_query = {"match_all": {}}
    response = get_data_from_elastic(elk_server=elk_server,
                                     elastic_index=elastic_index,
                                     elastic_query=elastic_query,
                                     dict_to_flat=dict_to_flat,
                                     use_default_fields=use_default_fields)
    return response


def list_to_or_query(value_list: list, parameter_name: str, key_name: str= 'match'):
    """
    Parses list of values to "should" query

    :param value_list: list of values
    :param parameter_name: match or term parameter
    :param key_name: either match or term
    :return: query dict
    """
    if len(value_list) == 1:
        return {key_name: {parameter_name: str(value_list[0])}}
    elif len(value_list) > 1:
        return {'bool': {'should': [{key_name: {parameter_name: str(single_value)}} for single_value in value_list]}}
    else:
        return None


class BoolQueryKey(Enum):
    """
    Keywords for multiple options when query from elastic
    """
    MUST = "must"
    SHOULD = "should"
    FILTER = 'filter'
    MUST_NOT = 'must_not'


def dict_to_and_or_query(value_dict: dict,
                         value_list: list = None,
                         key_name: str = 'match',
                         elastic_keywords: list = None,
                         bool_key: BoolQueryKey = BoolQueryKey.MUST):
    """
    Parses dictionary to query

    :param value_list:
    :param bool_key: specify "AND" ("must") or "OR" ("should")
    :param value_dict: parameter value dictionary
    :param key_name: term or match
    :param elastic_keywords: list of elastic keywords (necessary if full queries are provided)
    :return: elastic query
    """
    elastic_keywords = elastic_keywords or ELASTIC_KEYWORDS
    output = []
    queries = {}
    for dict_key, dict_value in value_dict.items():
        if dict_value is not None:
            if dict_key in [c.value for c in BoolQueryKey]:
                queries[dict_key] = dict_value
            elif dict_key in elastic_keywords:
                output.append({dict_key: dict_value})
            else:
                if isinstance(dict_value, list):
                    output.append(list_to_or_query(value_list=dict_value, parameter_name=dict_key, key_name=key_name))
                else:
                    output.append({key_name: {dict_key: str(dict_value)}})
    if isinstance(value_list, list) and len(value_list) >= 1:
        output.extend(value_list)
    if len(output) > 0:
        if bool_key.value in queries.keys():
            new_out = queries.get(bool_key.value)
            new_out = [new_out] if not isinstance(new_out, list) else new_out
            queries[bool_key.value] = [*new_out, *output]
        else:
            queries[bool_key.value] = output
    return {"bool": queries}


def search_key_from_dict(input_dict, keys: list, key: str = None):
    """
    Gets payload from query

    :param input_dict: input query
    :param keys: keywords to search
    :param key: return key
    :return: payload
    """
    if isinstance(input_dict, dict):
        for single_key in keys:
            if single_key in input_dict:
                return search_key_from_dict(input_dict=input_dict.get(single_key), keys=keys, key=single_key)
    return input_dict, key


def merge_queries(query_dict: dict, merge_dict: dict | list, default_key: BoolQueryKey = BoolQueryKey.MUST):
    """
    Merges two queries

    :param query_dict: query where to merge
    :param merge_dict: query to merge
    :param default_key: default key for merge
    :return: updated query
    """
    neglected_keys = ['bool', *[c.value for c in BoolQueryKey]]
    old_content, old_key = search_key_from_dict(input_dict=query_dict, keys=neglected_keys)
    new_content, new_key = search_key_from_dict(input_dict=merge_dict, keys=neglected_keys)
    old_key = old_key or default_key.value
    new_key = new_key or default_key.value
    if "bool" not in query_dict:
        query_dict = {"bool": {old_key: [old_content]}}
    if new_key in query_dict.get("bool", {}):
        content = query_dict["bool"][new_key]
        content = [content] if not isinstance(content, list) else content
        new_content = [new_content] if not isinstance(new_content, list) else new_content
        query_dict["bool"][new_key] = [*content, *new_content]
    else:
        if new_key != 'bool':
            query_dict["bool"][new_key] = new_content
    return query_dict


if __name__ == '__main__':
    elk_instance = ElkHandler(server=PY_ELASTICSEARCH_HOST)
    columns = DEFAULT_COLUMNS

    aceol_query = {
        "range": {
            "start_time": {
                "gte": "2024-01-01T00:00:00",
                "lte": "2025-01-01T00:00:00"
            }
        }
    }
    aceol_values = elk_instance.get_data(query=aceol_query,
                                         index=PY_SIZING_CURRENT_BALANCING_STATE_INDEX, ##[CHECK]
                                         fields=[['start_time'], ['end_time'], ['in_domain', 'name'], ['value', 'value']])
    print(aceol_values)
