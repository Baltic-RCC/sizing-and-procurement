import copy
import logging
import uuid

from http import HTTPStatus

import json
from enum import auto

import numpy
import pandas
import requests


from datetime import datetime

import config
from py.parsers.file_types import is_json, is_xml, is_xlsx, handle_xlsx, parse_input_by_type, is_zip_file, read_in_zip
from py.common.ref_constants import POINT_RESOLUTION_KEY, BUSINESS_TYPE_KEY, MRID_KEY, VALID_FROM_KEY, VALID_TO_KEY, \
    DOMAIN_NAME_KEY, TYPE_NAME_KEY, DIRECTION_NAME_KEY, \
    POINT_QUANTITY_KEY, PROCESS_TYPE_KEY, DIRECTION_CODE_KEY, DOMAIN_MRID_KEY, POINT_POSITION_KEY, MESSAGE_TYPE_KEY, \
    MARKET_PRODUCT_TYPE_KEY, AVAILABLE_POSITION_KEY, CURVE_TYPE_KEY, IN_DOMAIN_MRID_KEY, AVAILABLE_UNIT_KEY, \
    OUT_DOMAIN_MRID_KEY, SERIES_VALID_TO_KEY, SERIES_VALID_FROM_KEY
from py.common.to_elastic_logger import initialize_custom_logger
from py.data_classes.xml.balancing_market_document import BALANCING_XML_DOCUMENT_NAME, BALANCING_XML_TO_RESULT_MAP
from py.data_classes.xml.base_xml import XML_TO_RESULT_MAP
from py.data_classes.xml.capacity_document import CAPACITY_DOCUMENT_NAME, CAPACITY_XML_TO_RESULT_MAP
from py.data_classes.xml.merit_order_list_document import MERIT_ORDER_DOCUMENT_NAME, MERIT_XML_TO_RESULT_MAP
from py.data_classes.xml.reserve_bid_document import RESERVE_BID_DOCUMENT_NAME, RESERVE_XML_TO_RESULT_MAP
from py.handlers.elastic_handler import (PY_PROCUREMENT_ATC_INDEX, PY_PROCUREMENT_NCPB_INDEX,
                                         PY_SIZING_CURRENT_BALANCING_STATE_INDEX)
from py.common.config_parser import parse_app_properties
from py.parsers.json_to_calculation_result import delete_columns
from py.parsers.xml_to_calculation_result import generate_ids_from_dict, parse_and_filter_xml_to_dataframe
from py.handlers.elastic_handler import send_dictionaries_to_elastic, PY_ELASTICSEARCH_HOST
from py.procurement.constants import DEFAULT_AREA
from py.data_classes.enums import NameValueOfEnum, FlowDirectionType, BusinessType, ProcessType, MessageType, \
    MarketProductType
from py.common.functions import escape_empty_or_none, \
    calculate_start_and_end_date, convert_input, parse_to_type, unpack_dict_to_lists, key_exists, update_dict_values
from py.common.df_functions import parse_dataframe_to_nested_dict
from py.common.time_functions import TRANSPARENCY_DATETIME_FORMAT, ENTSOE_DATETIME_FORMAT, convert_datetime_to_string, \
    convert_datetime_to_string_utc, get_time_intervals

logger = logging.getLogger(__name__)
parse_app_properties(globals(), config.paths.config.transparency_data)

TRANSPARENCY_DATA_BASE_URL = TRANSPARENCY_ADDRESS
ENTSOE_DATA_BASE_URL = ENTSOE_ADDRESS
PY_SECURITY_TOKEN = SECURITY_TOKEN
PAGING_KEY = 'PAGING'
PAGE_KEY = 'page'
PARAM_KEY = 'parameters'
START_DATE_KEY = 'TRANSPARENCY_START_DATE'
END_DATE_KEY = 'TRANSPARENCY_END_DATE'
CONF_DOMAIN_KEY = 'DOMAIN'
CONF_DOMAIN_OUT_KEY = 'FROM_DOMAIN'
CONF_DOMAIN_TO_KEY = 'TO_DOMAIN'
ENTSOE_PAGING_KEY = 'offset'
ATC_COUNTRIES = [['Estonia', 'Latvia'], ['Latvia', 'Lithuania']]
ATC_TRANSPARENCY_QUERIES = convert_input(ATC_QUERIES)
BID_TRANSPARENCY_QUERIES = convert_input(BID_QUERIES)
IMBALANCE_TRANSPARENCY_QUERIES = convert_input(IMBALANCE_QUERIES)
QUERY_PARAMETERS = convert_input(QUERY_PARAMETERS)
MAX_RETRY = escape_empty_or_none(MAX_RETRY)

TRANSPARENCY_START_DATE = escape_empty_or_none(TRANSPARENCY_START_DATE)
TRANSPARENCY_END_DATE = escape_empty_or_none(TRANSPARENCY_END_DATE)
TRANSPARENCY_OFFSET = escape_empty_or_none(TRANSPARENCY_OFFSET)
TRANSPARENCY_TIME_DELTA = escape_empty_or_none(TRANSPARENCY_TIME_DELTA)
FROM_DOMAIN = escape_empty_or_none(FROM_DOMAIN)
TO_DOMAIN = escape_empty_or_none(TO_DOMAIN)
DOMAIN = escape_empty_or_none(DOMAIN)

PAGING_START = parse_to_type(PAGING_START, int)
PAGING_STEP = parse_to_type(PAGING_STEP, int)
PAGING_STOP = parse_to_type(PAGING_STOP, int)
# These are for ATC xmls coming from ENTSOE. Create additional mappings if new type of xml is coming

PY_INDEX_KEYS = convert_input(INDEX_KEYS)
PUBLICATION_DOCUMENT_NAME = 'Publication_MarketDocument'
PUBLICATION_XML_TO_RESULT_MAP = {'Period.resolution': POINT_RESOLUTION_KEY,
                                 'Point.position': AVAILABLE_POSITION_KEY,
                                 'Point.quantity': POINT_QUANTITY_KEY,
                                 # 'Point.secondaryQuantity': PERCENTAGE_VALUE_KEY,
                                 'TimeSeries.businessType': BUSINESS_TYPE_KEY,
                                 'TimeSeries.curveType': CURVE_TYPE_KEY,
                                 'TimeSeries.mRID': MRID_KEY,
                                 'TimeSeries.in_Domain.mRID': IN_DOMAIN_MRID_KEY,
                                 'TimeSeries.quantity_Measure_Unit.name': AVAILABLE_UNIT_KEY,
                                 'TimeSeries.out_Domain.mRID': OUT_DOMAIN_MRID_KEY,
                                 # 'TimeSeries.product': PRODUCT_KEY,
                                 # 'TimeSeries.secondary_Measurement_Unit.name': PERCENTAGE_UNIT_KEY,
                                 # 'domain.mRID': LFC_BLOCK_MRID_KEY,
                                 'Period.timeInterval.end': SERIES_VALID_TO_KEY,
                                 'Period.timeInterval.start': SERIES_VALID_FROM_KEY,
                                 'utc_start': VALID_FROM_KEY,
                                 'utc_end': VALID_TO_KEY}
PUBLICATION_XML_TO_RESULT_MAP = {**XML_TO_RESULT_MAP, **PUBLICATION_XML_TO_RESULT_MAP}
updates = {'available': 'Point', 'value': 'quantity',
           'percentage_level': 'Secondary_point',
           'area_Domain.mRID': DOMAIN_MRID_KEY,
           'Bid_TimeSeries.price_Measurement_Unit.name': 'Bid_TimeSeries.price_Measure_Unit.name',
           'Bid_TimeSeries.quantity_Measurement_Unit.name': 'Bid_TimeSeries.quantity_Measure_Unit.name',
           '_series_valid_to': 'series_valid_to',
           '_series_valid_from': 'series_valid_from'}
add_ons = {'Bid_TimeSeries.price_Measure_Unit.name': 'Secondary_point.measurement_unit',
           'Bid_TimeSeries.quantity_Measure_Unit.name': 'Point.measurement_unit'}

DOCUMENT_MAPPINGS = {BALANCING_XML_DOCUMENT_NAME: update_dict_values(BALANCING_XML_TO_RESULT_MAP, updates),
                     MERIT_ORDER_DOCUMENT_NAME: update_dict_values(MERIT_XML_TO_RESULT_MAP, updates),
                     RESERVE_BID_DOCUMENT_NAME: update_dict_values(RESERVE_XML_TO_RESULT_MAP, updates, add_ons),
                     CAPACITY_DOCUMENT_NAME: update_dict_values(CAPACITY_XML_TO_RESULT_MAP, updates),
                     PUBLICATION_DOCUMENT_NAME: update_dict_values(PUBLICATION_XML_TO_RESULT_MAP, updates)}

BTD_ID_TYPES = {MESSAGE_TYPE_KEY: {'VOLUMES': MessageType.RESERVE_TENDER_DOCUMENT.value,
                                   'PROCURED': MessageType.ACQUIRING_SYSTEM_OPERATOR_RESERVE_SCHEDULE.value,
                                   'DEMAND': MessageType.RESERVE_TENDER_DOCUMENT.value,
                                   'BALANCING': MessageType.IMBALANCE_VOLUME.value},
                BUSINESS_TYPE_KEY: {'VOLUMES': BusinessType.OFFER.value,
                                    'PROCURED': BusinessType.PROCURED_CAPACITY.value,
                                    'DEMAND': BusinessType.NEED.value,
                                    'BALANCING': BusinessType.AREA_CONTROL_ERROR.value},
                PROCESS_TYPE_KEY: {'RR': ProcessType.REPLACEMENT_RESERVE.value,
                                   'BALANCING': ProcessType.REALISED.value,
                                   'FCR': ProcessType.FREQUENCY_CONTAINMENT_RESERVE.value,
                                   'AFRR': ProcessType.AUTOMATIC_FREQUENCY_RESTORATION_RESERVE.value,
                                   'MFRR': ProcessType.MANUAL_FREQUENCY_RESTORATION_RESERVE.value},
                MARKET_PRODUCT_TYPE_KEY: {'SA': MarketProductType.MFRR_PRODUCT_FOR_SCHEDULED_ACTIVATION.value,
                                          'DA': MarketProductType.MFRR_PRODUCT_FOR_SCHEDULED_DIRECT_ACTIVATION.value}}
BTD_COLUMN_NAMES = {'from': VALID_FROM_KEY,
                    'to': VALID_TO_KEY,
                    'res': POINT_RESOLUTION_KEY,
                    'group_level_0': DOMAIN_NAME_KEY,
                    'group_level_1': TYPE_NAME_KEY,
                    'label': DIRECTION_NAME_KEY,
                    'values': POINT_QUANTITY_KEY}
BTD_DATA_FIELDS = {'id': MRID_KEY, 'measurement_unit': 'Point.measurement_unit'}


class DownloadType(NameValueOfEnum):
    """
    Enum fo selecting download type
    """
    NONE = auto()
    BIDS = auto()
    CURRENT_BALANCING_STATE = auto()
    ATC = auto()
    ALL = auto()


BIDS_TABLES = {PY_PROCUREMENT_NCPB_INDEX: BID_TRANSPARENCY_QUERIES}
VOLUMES_TABLES = {PY_SIZING_CURRENT_BALANCING_STATE_INDEX: IMBALANCE_TRANSPARENCY_QUERIES}
ATC_TABLES = {PY_PROCUREMENT_ATC_INDEX: ATC_TRANSPARENCY_QUERIES}


def get_http_request(base_address: str, payload: dict, max_retry: str = MAX_RETRY):
    """
    Method for getting data with http request. Currently, no error handling is done

    :param base_address: address for http request
    :param payload: parameters for the request
    :param max_retry: maximum attempt of api calls tries
    :return: content if response was 200 (OK) else print the content to screen
    """
    if max_retry is None:
        max_try = 3
    else:
        max_try = int(max_retry)
    for attempt in range (1, max_try +1):
        try:
            response = requests.get(url=base_address, params=payload)
            logger.info(f'Query: {response.request.url}') #debug
            if response.status_code == 200:
                logger.info(f'number of attempts: {attempt}')
                return response.content, response.status_code
            else:
                logger.info(f'Attempt failed {response.status_code}') #debug

        except requests.RequestException as e:
            logger.info(f'query failed: {e}') #debug

    return None, response.status_code


def btd_data_to_dataframe(headers: list, time_series: list):
    """
    For the transparency data pads headers with values from time series

    :param headers: column names
    :param time_series: values
    :return: pandas dataframe
    """
    output = []
    for x in time_series:
        for y in headers:
            new_y = copy.deepcopy(y)
            for k, v in x.items():
                if not isinstance(v, list):
                    new_y[k] = v
                else:
                    new_y[k] = v[y.get('index')]
            output.append(new_y)
    return pandas.DataFrame(output)


def get_match_from_id_string(id_string, mappings: dict, delimiter: str = '_'):
    """
    For translating id string to components based on mappings

    :param id_string: input string
    :param mappings: key value pairs
    :param delimiter: use this to divide the string to components
    :return: matches
    """
    if id_string is None or mappings is None:
        return None
    components = id_string.upper().split(delimiter)
    components = components[0] if len(components) == 1 else components
    for k, v in mappings.items():
        if str(k).upper() in components:
            return v
    return None


def get_matches_from_id_string(id_string, mappings: dict, delimiter: str = '_'):
    """
    For translating id string to components based on mappings

    :param id_string: input string
    :param mappings: key value pairs
    :param delimiter: use this to divide the string to components
    :return: matches
    """
    outputs = {}
    if id_string is None or mappings is None:
        return outputs
    for k, v in mappings.items():
        if isinstance(v, dict):
            v_val = get_match_from_id_string(id_string=id_string, mappings=v, delimiter=delimiter)
            if v_val is not None:
                outputs[k] = v_val
    return outputs


def add_additional_fields(input_data: pandas.DataFrame, overwrite: bool = True, fields: dict = None):
    """
    Adds additional fields to dataframe

    :param fields: dictionary of field names and values
    :param input_data: input dataframe
    :param overwrite: if field exist then overwrite or bypass it
    :return: updated dictionary
    """
    if isinstance(fields, dict):
        for k, v in fields.items():
            if isinstance(k, str) and isinstance(v, str):
                if not overwrite and k in input_data.columns:
                    continue
                input_data[k] = v
    return input_data

def parse_btd_json(input_data,
                   column_names: dict = None,
                   id_mapping: dict = None,
                   data_mapping: dict = None,
                   add_data: dict = None,
                   **kwargs):
    """
    Parses json to dictionaries

    :param add_data: dictionary with additional constants that need to be added
    :param data_mapping: translate fields from data to objects
    :param id_mapping: translate id field in data to components
    :param column_names: translate index column names
    :param input_data: json data
    :return: list of TransparencyDataPoints
    """
    direction_mapping = {'Upward': 'UP', 'Downward': 'DOWN', 'Symetric': 'UP_AND_DOWN'}
    data_mapping = BTD_DATA_FIELDS if data_mapping is None else data_mapping
    column_names =  BTD_COLUMN_NAMES if column_names is None else column_names
    id_mapping = BTD_ID_TYPES if id_mapping is None else id_mapping

    json_data = {}
    if isinstance(input_data, bytes):
        json_data = json.loads(input_data)
    data = json_data.get('data', {})
    btd_df = btd_data_to_dataframe(headers=data.get('columns', []), time_series=data.get('timeseries', []))
    btd_df = delete_columns(input_dataframe=btd_df, columns_delete=['index', 'col'], delete_empty=True)
    btd_df = btd_df.rename(columns=column_names)
    if DIRECTION_NAME_KEY in btd_df.columns:
        btd_df[DIRECTION_NAME_KEY] = btd_df[DIRECTION_NAME_KEY].replace(direction_mapping)
        btd_df[DIRECTION_CODE_KEY] = btd_df[DIRECTION_NAME_KEY].apply(lambda x: FlowDirectionType.value_of(x).value)
    domains = {x.name: x.mRID for x in DEFAULT_AREA}
    btd_df[DOMAIN_MRID_KEY] = btd_df[DOMAIN_NAME_KEY].apply(lambda x: domains.get(x))
    btd_df[POINT_POSITION_KEY] = 1
    for k, v in data_mapping.items():
        btd_df[v] = data.get(k)
    id_items = get_matches_from_id_string(id_string=data.get('id'), mappings=id_mapping)
    for k, v in id_items.items():
        btd_df[k] = v
    btd_df = add_additional_fields(input_data=btd_df, overwrite=True, fields=add_data)
    if TYPE_NAME_KEY in btd_df.columns and PROCESS_TYPE_KEY in id_mapping.keys():
        pt = btd_df[TYPE_NAME_KEY].apply(lambda x: get_match_from_id_string(id_string=x,
                                                                            mappings=id_mapping[PROCESS_TYPE_KEY],
                                                                            delimiter=' '))
        if PROCESS_TYPE_KEY in btd_df.columns:
            btd_df[PROCESS_TYPE_KEY] = btd_df[PROCESS_TYPE_KEY].fillna(pt)
        else:
            btd_df[PROCESS_TYPE_KEY] = pt
    parsed_data = parse_dataframe_to_nested_dict(input_dataframe=btd_df)
    if parsed_data is not None:
        return generate_ids_from_dict(input_data=parsed_data, id_fields=PY_INDEX_KEYS)
    return None


def xml_to_elastic(input_data, mappings: dict = None, overwrite_col: bool = True, add_data: dict = None,  **kwargs):
    """
    Parses xml to dictionaries where base is point and returns list of dicts where key is id for elastic and
    data is payload

    :param add_data: dictionary with additional constants that need to be added
    :param overwrite_col: when mapping outcome contains already existing column then either to overwrite or keep it
    :param mappings: dictionary of document type field mapping
    :param input_data: xml as bytes
    :return: list of dictionaries or None
    """
    parsed_data = None
    mappings = mappings or DOCUMENT_MAPPINGS
    try:
        output, doc_type = parse_and_filter_xml_to_dataframe(input_data=input_data)
        output = add_additional_fields(input_data=output, overwrite=True, fields=add_data)
        if not output.empty:
            column_mapping = mappings.get(doc_type[0]) if isinstance(doc_type, list) and len(doc_type) == 1 else None
            if column_mapping is not None:
                vals = [v for k, v in column_mapping.items() if k != v]
                drop = [x for x in output.columns.tolist() if x in vals]
                if len(drop) > 0:
                    if not overwrite_col:
                        output = output.drop(columns=drop)
                    else:
                        drop = {x: x + "_2" for x in drop}
                        output = output.rename(columns=drop)
                output = output.rename(columns=column_mapping)
            else:
                raise Exception(f'No mapping was found for {doc_type}')
            parsed_data = parse_dataframe_to_nested_dict(input_dataframe=output)

    except AttributeError as ae:
        pass
        # logger.warning(f"Cannot parse xml to point dataframe: {ae}")
    except Exception as ex:
        logger.warning(f"Unknown error occurred when parsing xml {ex}")
    if parsed_data is not None:
        generated = generate_ids_from_dict(input_data=parsed_data, id_fields=PY_INDEX_KEYS)
        if len(generated.keys()) != len(parsed_data):
            raise Exception(f"Error in id")
        return generated
    return None



def get_borders(borders: list = None, area_list: list = None):
    """
    Gets tuples of mRIDs for borders if exists

    :param area_list: List of EICArea
    :param borders: list of tuples of names for the borders
    :return: list of tuples of mRID for borders
    """
    if borders is None:
        borders = ATC_COUNTRIES
    if area_list is None:
        area_list = DEFAULT_AREA
    mrid_pairs = []
    for country_pair in borders:
        m_rids = [x.mRID for x in area_list for y in country_pair if x.name.lower() == y.lower()]
        mrid_pairs.extend([(x, y) for x in m_rids for y in m_rids if x != y])
    return mrid_pairs


def str_to_variable(input_dict: dict, mappings: dict = None):
    """
    Maps strings in input dictionary represented with <> to corresponding variables. Checks mapping first
    and if not found then goes to global variables

    :param input_dict: dictionary where to replace the strings
    :param mappings: additional key, value pairs
    :return: updated dictionary if all necessary were replaced, None otherwise
    """
    output_dict = copy.deepcopy(input_dict)
    mappings = {} if mappings is None else mappings
    mappings = {str(k).strip('<>'): v for k, v in mappings.items()}
    for k, v in output_dict.items():
        if isinstance(v, dict):
            new_v = str_to_variable(input_dict=v, mappings=mappings)
            if new_v is None:
                return None
            output_dict[k] = str_to_variable(input_dict=v, mappings=mappings)
        else:
            if str(v).startswith('<') and str(v).endswith('>'):
                cleaned_v = str(v).strip('<>')
                try:
                    v_val = globals()[cleaned_v] if mappings.get(cleaned_v) is None else mappings.get(cleaned_v)
                    if v_val is not None:
                        output_dict[k]  = v_val
                    else:
                        return None
                except KeyError:
                    return None
    return output_dict


def update_query_by_parameters(query_dict: dict, parameter_list: list | dict):
    """
    Generates set of queries based on the input query and parameter combinations

    :param query_dict: query dictionary
    :param parameter_list: list of parameter combination
    :return: list of queries
    """
    output = []
    parameter_list = [parameter_list] if not isinstance(parameter_list, list) else parameter_list
    for parameter_set in parameter_list:
        single_query = str_to_variable(input_dict=query_dict, mappings=parameter_set)
        if single_query is not None:
            output.append(single_query)
    return [x for n, x in enumerate(output) if output.index(x) == n]


def generate_queries(elastic_queries: dict, parameter_list: list, paging_key: str, paging_list: list):
    """
    Generates queries. For each query injects the parameters and adds pagination if applicable

    :param elastic_queries: dictionary consisting of elastic index name with list of queries
    :param parameter_list: list of parameter combinations (dictionary)
    :param paging_key: key for detecting pagination
    :param paging_list: list of pagination parameters
    :return: generated dictionary of elastic index name with list of queries
    """
    for table_name in elastic_queries:
        query_list = elastic_queries.get(table_name, [])
        query_list = [query_list] if not isinstance(query_list, list) else query_list
        new_queries = []
        for single_query in query_list:
            results = update_query_by_parameters(query_dict=single_query, parameter_list=parameter_list)
            buffer = []
            for result_value in results:
                if key_exists(result_value, paging_key):
                    new_queries.append(update_query_by_parameters(query_dict=result_value, parameter_list=paging_list))
                else:
                    buffer.append(result_value)
            if len(buffer) > 0:
                new_queries.append(buffer)
        elastic_queries[table_name] = new_queries
    return elastic_queries


def query_data(address: str,
               parameters: dict,
               index_name: str,
               date_format: str = TRANSPARENCY_DATETIME_FORMAT,
               parsers: dict = None,
               elastic_host: str = PY_ELASTICSEARCH_HOST):
    """
    Queries data from address by parameters, parses it and stores output to elastic index

    :param address: address of a page
    :param parameters: query parameters
    :param index_name: index where to store the data
    :param date_format: date format for parsing dates to strings
    :param parsers: dictionary of type checks and type parsers
    :param elastic_host: address of elastic host
    :return:
    """
    parsers = parsers or {is_xml: xml_to_elastic, is_json: parse_btd_json, is_xlsx: handle_xlsx}
    for k, v in parameters.items():
        if isinstance(v, datetime):
            v = convert_datetime_to_string(input_value=v, output_format=date_format)
            parameters[k] = v
    response, status_code = get_http_request(base_address=address, payload=parameters)
    received_files = read_in_zip(zip_content=response) if is_zip_file(response) else [response]
    received_length = 0
    for received in received_files:
        content = parse_input_by_type(input_data=received, type_dict=parsers, add_data={'source': address})
        logger.info("Content parsed")
        if content is not None and len(content) > 0:
            send_dictionaries_to_elastic(input_list=content,elastic_index_name=index_name,elastic_server=elastic_host)
            logger.info("Content sent to elastic")
            received_length = received_length + len(content)
    return received_length, status_code


def download_data(download_dict: dict,
                  date_forms: dict = None,
                  page_length: int = PAGING_STEP,
                  page_key: str = ENTSOE_PAGING_KEY):
    """
    Downloads data

    :param page_key: key to detect paging in parameters
    :param page_length: page length (to cut off querying if length is less)
    :param download_dict: dictionary with elastic index and query pairs
    :param date_forms: dictionary for date formats for different pages
    :return:
    """
    date_forms = date_forms or {'entsoe': ENTSOE_DATETIME_FORMAT, 'baltic': TRANSPARENCY_DATETIME_FORMAT}
    for i, data_index in enumerate(download_dict.keys()):
        logger.info(f"Downloading to {data_index} ({i + 1}/{len(download_dict.keys())})")
        queries = downloads.get(data_index, [])
        queries = [queries] if not isinstance(queries, list) else queries
        q_length = len(queries)
        for j, query_set in enumerate(queries):
            i_length = 0
            s_length = len(query_set)
            for m, query in enumerate(query_set):
                page = query.get(PAGE_KEY)
                params = query.get(PARAM_KEY, {})
                date_form = next(iter([v for k, v in date_forms.items() if str(k).lower() in str(page).lower()]), None)
                print(params) #DEBUG
                r_len, code = query_data(address=page, parameters=params, index_name=data_index, date_format=date_form)
                i_length = i_length + r_len
                p_str =(f"{page}: {code}.Got {r_len} values, {i_length} total.")
                if page_key in params.keys() and r_len < page_length:
                    logger.info(p_str + ' Last page')
                    break
                else:
                    logger.info(p_str)


if __name__ == '__main__':
    # Whether to send logs to elastic
    initialize_custom_logger(extra_fields={"Job": "Transparency Data", "Job_id": str(uuid.uuid4())})
    # Calculate query start and end based on parameters provided
    start_date, end_date = calculate_start_and_end_date(start_date_time=TRANSPARENCY_START_DATE,
                                                        end_date_time=TRANSPARENCY_END_DATE,
                                                        offset=TRANSPARENCY_OFFSET,
                                                        time_zone='CET',
                                                        time_delta=TRANSPARENCY_TIME_DELTA)
    # Divide start and end date to intervals
    download_dates = get_time_intervals(start_date_value=start_date, end_date_value=end_date, time_delta=QUERY_INTERVAL)
    # Create or update query parameters
    query_parameters = {} if QUERY_PARAMETERS is None else copy.deepcopy(QUERY_PARAMETERS)
    query_parameters[(START_DATE_KEY, END_DATE_KEY)] = download_dates
    query_parameters[CONF_DOMAIN_KEY] = DOMAIN or [x.mRID for x in DEFAULT_AREA]
    domain_pairs = [(FROM_DOMAIN, TO_DOMAIN)] if FROM_DOMAIN and TO_DOMAIN else (
        get_borders(borders=ATC_COUNTRIES, area_list=DEFAULT_AREA))
    query_parameters[(CONF_DOMAIN_OUT_KEY, CONF_DOMAIN_TO_KEY)] = domain_pairs
    paging_key_variable = '<' + PAGING_KEY + '>'
    query_parameters[PAGING_KEY] = [paging_key_variable]
    query_parameters = unpack_dict_to_lists(query_parameters)
    # Generate paging parameters
    paging_values = numpy.arange(PAGING_START, PAGING_STOP + PAGING_STEP, PAGING_STEP).tolist()
    paging_data = unpack_dict_to_lists({PAGING_KEY: paging_values})
    # Parse and load in download types
    downloads = {}
    download_type = escape_empty_or_none(TRANSPARENCY_DOWNLOAD_TYPE)
    download_type = DownloadType.value_of(download_type) if download_type else DownloadType.ALL
    logger.info(f"Downloading {download_type.name} "
                f"from {convert_datetime_to_string_utc(start_date)} to {convert_datetime_to_string_utc(end_date)}")
    if download_type == DownloadType.BIDS or download_type == DownloadType.ALL:
        downloads = {**downloads, **BIDS_TABLES}
    if download_type == DownloadType.CURRENT_BALANCING_STATE or download_type == DownloadType.ALL:
        downloads = {**downloads, **VOLUMES_TABLES}
    if download_type == DownloadType.ATC or download_type == DownloadType.ALL:
        downloads = {**downloads, **ATC_TABLES}
    # Generate queries
    downloads = generate_queries(elastic_queries=downloads,
                                 paging_key=paging_key_variable,
                                 parameter_list=query_parameters,
                                 paging_list=paging_data)
    # Download data by queries
    download_data(download_dict=downloads)
    print("Done")
