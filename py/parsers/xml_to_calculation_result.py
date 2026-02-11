import copy
import logging
import os

import sys
import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import uuid4

from lxml import etree

import pandas

import config
from py.data_classes.results.result_functions import ATC_XML_RESULT_TYPES, BID_XML_RESULT_TYPES
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_bid import CalculationResultBid
from py.data_classes.task_classes import ProcurementCalculationTask
from py.handlers.elastic_handler import ElkHandler as Elastic, PY_ELASTICSEARCH_HOST
from py.common.config_parser import parse_app_properties
from py.common.functions import check_and_create_the_folder_path, \
    convert_input, get_file_path_from_root_by_name
from py.common.df_functions import parse_dataframe_to_nested_dict
from py.common.time_functions import convert_string_to_datetime, parse_duration
from py.procurement.procurement_common import get_task_from_environment

logger = logging.getLogger(__name__)

parse_app_properties(globals(), config.paths.config.atc_input)

PY_PROD_ATC_INDEX_KEYS = convert_input(PROD_ATC_INDEX_KEYS)
ELASTIC_CHECK_TYPE_INCLUDED = []
ELASTIC_CHECK_TYPE_EXCLUDED = ['position', 'quantity', 'number']
ELASTIC_CHECK_DESIRED_TYPE = 'str'


def add_rdf_object_to_list(single_rdf_object,
                           list_of_items,
                           object_id,
                           parent_id,
                           id_triggered: str = None,
                           prefix: str = None,
                           use_prefix: bool = False,
                           prefix_delimiter: str = '.',
                           recursion_counter: int = 1,
                           max_recursion: int = 10):
    """
    Recursively reads rdf objects from graph to list

    :param single_rdf_object: graph element to parse
    :param list_of_items: current list of parsed elements
    :param object_id: id to be used for the fields in rdf_object
    :param parent_id: parent id to be used for the fields ind rdf_object
    :param id_triggered: specify field name for which new id is triggered
    :param prefix: prefix to indicate the parent
    :param use_prefix: if true then parent field name will be added to the fields as prefix
    :param prefix_delimiter: character to be used for separating prefix and field name
    :param recursion_counter: counter for the recursion depth
    :param max_recursion: guard when graph gets too deep
    """
    attribute_object_key = None
    attribute_object_value = None
    if single_rdf_object.attrib:
        attribute_object_key = single_rdf_object.attrib.keys()[0]
        attribute_object_value = single_rdf_object.attrib[attribute_object_key]

    object_value, object_key = single_rdf_object.tag.split('}')
    if single_rdf_object.text:
        if prefix:
            object_key = prefix + prefix_delimiter + object_key
        object_value = single_rdf_object.text
    else:
        # If needed, keep all nested children under one ID
        if not id_triggered or id_triggered == object_key:
            parent_id = object_id
            object_id = str(uuid4())
        object_value = object_key
        object_key = 'Type'
        # For the sake cleanliness escape prefix for those which are used for the id
        if use_prefix and object_value != id_triggered:
            prefix = object_value
        # list_of_items.append((object_id, "parent", old_object_id, parent_id))
    if attribute_object_key and attribute_object_value:
        attribute_object_key = object_key + prefix_delimiter + attribute_object_key
        list_of_items.append((object_id, attribute_object_key, attribute_object_value, parent_id))
    list_of_items.append((object_id, object_key, object_value, parent_id))
    # Just in case escape from recursion if going too deep
    if recursion_counter > max_recursion:
        return list_of_items
    for rdf_child in single_rdf_object.iterchildren():
        list_of_items = add_rdf_object_to_list(single_rdf_object=rdf_child,
                                               list_of_items=list_of_items,
                                               object_id=object_id,
                                               parent_id=parent_id,
                                               id_triggered=id_triggered,
                                               use_prefix=use_prefix,
                                               prefix_delimiter=prefix_delimiter,
                                               prefix=prefix,
                                               recursion_counter=recursion_counter + 1,
                                               max_recursion=max_recursion)
    return list_of_items


class RelativeKind(Enum):
    PARENT = auto()
    CHILD = auto()

def get_relatives(input_data: pandas.DataFrame,
                  type_id: pandas.DataFrame,
                  data_to_pivot: pandas.DataFrame,
                  recursion_depth: int = 10,
                  recursion_counter: int = 0,
                  relative_kind: RelativeKind = RelativeKind.PARENT,
                  get_type_ids: bool = True
                  ):
    """
    Currently gets parents for parents (moving from branch towards stem)

    :param input_data: original triplets data
    :param type_id: slice of triplets to which parents are searched
    :param data_to_pivot: slice of triplets containing the data for tableview
    :param recursion_depth: how far can go with triplets
    :param recursion_counter: current recursion counter
    :param relative_kind: PARENT:
    :param get_type_ids:
    :return: updated data_to_pivot
    """
    if recursion_counter >= recursion_depth:
        return data_to_pivot
    relative_data = pandas.DataFrame()
    if relative_kind == RelativeKind.PARENT:
        relative_data = pandas.merge(type_id[['ID', 'PARENT_ID']], input_data, right_on="ID", left_on="PARENT_ID",
                                   suffixes=('', '_duplicate'))
    elif relative_kind == RelativeKind.CHILD:
        relative_data = pandas.merge(type_id[['ID', 'PARENT_ID']], input_data, right_on='PARENT_ID', left_on='ID',
                                     suffixes=('', '_duplicate'))

    if not relative_data.empty:
        if relative_kind == RelativeKind.PARENT:
            relative_data['PARENT_ID'] = relative_data['PARENT_ID_duplicate']
        if get_type_ids:
            type_fields = relative_data[relative_data['KEY'] == 'Type']
            type_fields.loc[:, 'KEY'] = type_fields['VALUE'].astype(str) + '_ID'
            type_fields.loc[:, 'VALUE'] = type_fields['ID_duplicate']
            relative_data = pandas.concat([relative_data, type_fields])
        relative_data.drop(relative_data.filter(regex='_duplicate$').columns, axis=1, inplace=True)
        parent_ids = relative_data[['ID', 'PARENT_ID']].drop_duplicates(subset=['ID', 'PARENT_ID'], keep="first")
        data_to_pivot = pandas.concat([data_to_pivot, relative_data])
        if relative_kind != RelativeKind.CHILD:
            data_to_pivot = get_relatives(input_data=input_data,
                                          type_id=parent_ids,
                                          data_to_pivot=data_to_pivot,
                                          recursion_depth=recursion_depth,
                                          recursion_counter=recursion_counter+1)
    return data_to_pivot


def type_tableview_in_depth(input_data,
                            type_name,
                            string_to_number=True,
                            get_parent_levels: int = 10,
                            get_child: bool = True):
    """
    Creates a table view of all objects of same type, with their parameters in columns. Borrowed from triplets
    library. Currently work in progress, but it takes the fields from 1st level parent also

    :param input_data: data as triplets (see rdf parsing)
    :param type_name: the value for which the table is created
    :param string_to_number: convert strings to number
    :param get_parent_levels: gets fields from parent element
    :param get_child: get children
    """
    # Get all ID-s of rows where Type == type_name
    type_id = input_data.query("VALUE == '{}' & KEY == 'Type'".format(type_name))
    type_parent_id = copy.deepcopy(type_id)
    type_parent_id.loc[:, 'KEY'] = type_parent_id['VALUE'] + '_PARENT_ID'
    type_parent_id.loc[:, 'VALUE'] = type_parent_id['PARENT_ID']
    if type_id.empty:
        # logger.warning('No data available for {}'.format(type_name))
        return None

    # Filter original data by found type_id data
    # There can't be duplicate ID and KEY pairs for pivot, but this will lose data on full model
    # DependantOn and other info, solution would be to use pivot table function.
    type_data = (pandas.merge(type_id[["ID"]], input_data, right_on="ID", left_on="ID")
                 .drop_duplicates(["ID", "KEY"]))
    if get_parent_levels and get_parent_levels > 0:
        type_data = get_relatives(input_data=input_data,
                                  type_id=type_id,
                                  data_to_pivot=type_data,
                                  recursion_depth=get_parent_levels,
                                  relative_kind=RelativeKind.PARENT,
                                  recursion_counter=0)
    # as children depend on ID, and we pivot on id then the depth will and remain to be 1
    if get_child:
        type_data = get_relatives(input_data=input_data,
                                  type_id=type_id,
                                  data_to_pivot=type_data,
                                  recursion_depth=1,
                                  relative_kind=RelativeKind.CHILD,
                                  recursion_counter=0)
    type_data = pandas.concat([type_data, type_parent_id])
    type_data = type_data.drop_duplicates(['ID', 'KEY'])

    # Convert form triplets to a table view all objects of same type
    data_view = type_data.pivot(index="ID", columns="KEY")["VALUE"]

    if string_to_number:
        # Convert to data type to numeric in columns that contain only numbers (for easier data usage later on)
        data_view_converted = data_view.apply(pandas.to_numeric, errors='coerce')
        data_view_converted = data_view_converted.fillna(data_view)
        return data_view_converted
    return data_view


def parse_time_series(time_series_path, prefix_delimiter: str = '.', is_path: bool = True):
    """
    Parses given xml to "RegisteredResource" type tableview

    :param time_series_path: path to the xml
    :param prefix_delimiter: set delimiter for the prefixes
    :param is_path: whether input is path
    """
    parser = etree.XMLParser(remove_comments=True, collect_ids=False, remove_blank_text=True)
    if (isinstance(time_series_path, str) and not is_path) or isinstance(time_series_path, bytes):
        parsed_xml = etree.fromstring(time_series_path, parser=parser)
    else:
        parsed_xml = etree.parse(time_series_path, parser=parser)

    file_id = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())

    try:
        rdf_object = parsed_xml.getroot()
        rdf_objects = rdf_object.iterchildren()
    except AttributeError:
        rdf_object = parsed_xml
        rdf_objects = parsed_xml.iterchildren()

    data_list = []
    # Add root
    _, object_value = rdf_object.tag.split('}')
    object_key = 'Type'
    data_list.append((file_id, object_key, object_value, parent_id))
    # first_id = str(uuid4())
    first_id = file_id
    for RDF_object in rdf_objects:

        data_list = add_rdf_object_to_list(single_rdf_object=RDF_object,
                                           list_of_items=data_list,
                                           object_id=first_id,
                                           use_prefix=True,
                                           prefix_delimiter=prefix_delimiter,
                                           # id_triggered='ReserveBidTimeSeries',
                                           parent_id=parent_id)

    time_series = pandas.DataFrame(data_list, columns=["ID", "KEY", "VALUE", "PARENT_ID"])
    return time_series


def time_for_position(input_time: datetime | str, duration : str, position: int | str, offset: int = 1):
    """
    Calculates time for position

    :param input_time: start time
    :param duration: duration of the position
    :param position: the number of position
    :param offset: 1: start, 0: end
    :return: datetime object
    """
    duration = parse_duration(duration)
    input_time = convert_string_to_datetime(input_time)
    position = max(int(position) - offset, 0)
    input_time = input_time + position * duration
    return input_time


def parse_triplets_to_point_tableview(input_data: pandas.DataFrame, date_time_format: str = '%Y-%m-%dT%H:%M:%S%z'):
    """
    Not a very polite way but it gets most of the data for the Point (ATC and bids)

    :param input_data: triplets
    :param date_time_format: for utc_start and utc_end columns
    :return: dataframe containing from Point to Document (and some side classes also)
    """
    period = 'Period'
    period_int = 'timeInterval'
    doc_stat = 'docStatus'
    # Get input data
    tableview = (type_tableview_in_depth(input_data=input_data, type_name='Point')
                 .reset_index().drop(columns=['ID', 'Type']))
    main_id_column = [x for x in tableview.columns.to_list() if str(x).endswith('Document_ID')]
    if len(main_id_column) == 1:
        main_id_column = main_id_column[0]
    else:
        main_id_column = None

    # Add timeIntervals to Periods
    if f"{period}_ID" in tableview.columns.to_list():

        time_intervals = type_tableview_in_depth(input_data=input_data, type_name=period_int,
                                                 get_parent_levels=0,
                                                 get_child=False).reset_index().drop(columns=['ID', 'Type'])
        tableview = tableview.merge(time_intervals.rename(columns={f'{period_int}_PARENT_ID': f'{period}_ID',
                                                                   f'{period_int}.start': f'{period}.{period_int}.start',
                                                                   f'{period_int}.end': f'{period}.{period_int}.end'}),
                                    on=f'{period}_ID')

    # Add general timeInterval
    unique_types = input_data[input_data['KEY'] == 'Type']
    unique_types = list(unique_types['VALUE'].unique())
    main_intervals = [x for x in unique_types if str(x).endswith('.timeInterval')]
    if len(main_intervals) == 1:
        main_int = main_intervals[0]
        doc_interval = type_tableview_in_depth(input_data=input_data, type_name=main_int,
                                               get_parent_levels=0,
                                               get_child=False).reset_index().drop(columns=['ID', 'Type'])
        if not doc_interval.empty and main_id_column:
            tableview = tableview.merge(doc_interval.rename(columns={f'{main_int}_PARENT_ID': main_id_column}),
                                        on=main_id_column)
    # Add doc status
    doc_stat_key = [x for x in unique_types if str(x).upper() == doc_stat.upper()]
    if len(doc_stat_key) == 1:
        doc_stat_key = doc_stat_key[0]
        doc_status = type_tableview_in_depth(input_data=input_data, type_name=doc_stat_key,
                                             get_parent_levels=0,
                                             get_child=False).reset_index().drop(columns=['ID', 'Type'])
        if not doc_status.empty and len(main_id_column) == 1:
            tableview = tableview.merge(doc_status.rename(columns={f'{doc_stat_key}_PARENT_ID': main_id_column[0]}),
                                        on=main_id_column[0])

    tableview.drop(tableview.filter(regex='_ID$').columns, axis=1, inplace=True)

    # calculate utc_start and utc_end
    tableview['utc_start'] = (tableview.apply(lambda x: time_for_position(input_time=x['Period.timeInterval.start'],
                                                                          duration=x['Period.resolution'],
                                                                          position=x['Point.position']), axis=1))
    tableview['utc_end'] = (tableview
                            .apply(lambda x: time_for_position(input_time=x['Period.timeInterval.start'],
                                                               duration=x['Period.resolution'],
                                                               position=x['Point.position'],
                                                               offset=0), axis=1))
    tableview['utc_start'] = pandas.to_datetime(tableview['utc_start'], utc=True).dt.strftime(date_time_format)
    tableview['utc_end'] = pandas.to_datetime(tableview['utc_end'], utc=True).dt.strftime(date_time_format)
    return tableview


def generate_ids_from_dict(input_data: list,
                           id_fields: list = PY_PROD_ATC_INDEX_KEYS,
                           key_delimiter: str = '-'):
    """
    Generates dictionary where key is custom id based on id_fields and value is previous dict
    (For storing data to elastic)

    :param input_data: list of (nested) dictionaries as elastic likes it
    :param id_fields: for nested structures use list (list of lists) for single level use strings (list of strings)
    :param key_delimiter: delimiter to be used to join the fields
    :return:
    """
    max_key_length = 512
    new_output = {}
    if not isinstance(id_fields, list):
        id_fields = [id_fields]
    for single_dict in input_data:
        id_composed = []
        for id_field in id_fields:
            if not isinstance(id_field, list):
                id_field = [id_field]
            output_value = single_dict
            for single_field in id_field:
                if isinstance(output_value, dict):
                    output_value = output_value.get(single_field)
            # Extend these but keep in mind to avoid special characters (dict, list, tuple)
            if isinstance(output_value, (int, float, complex)):
                output_value = str(output_value)
            if isinstance(output_value, str):
                id_composed.append(output_value)
        new_id_key = key_delimiter.join(id_composed)
        new_id_key = new_id_key[:max_key_length]
        new_output[new_id_key] = single_dict
    return new_output


def filter_dataframe(input_data: pandas.DataFrame, key_value_pairs: dict):
    """
    For changed filtering by multiple parameters

    :param input_data: input dataframe
    :param key_value_pairs: key value pairs for filtering (only equals is used)
    :return: filtered dataframe or empty one
    """
    for filter_key, filter_value in key_value_pairs.items():
        if input_data.empty:
            return  input_data
        input_data = input_data[input_data[filter_key] == filter_value]
    return input_data


def filter_list(input_data: list, included: list | str = None, excluded: list | str = None):
    """
    Filters list of strings based included and excluded keywords.
    Note that excluded overwrites included

    :param input_data: list of strings
    :param included: keywords that should be included (case-insensitive)
    :param excluded: keywords that should be excluded (case-insensitive)
    :return: filtered list
    """
    included = [included] if isinstance(included, str) else included
    excluded = [excluded] if isinstance(excluded, str) else excluded
    input_data = [x for x in input_data
                  if any(x for y in included if str(y).lower() in str(x).lower())] if included else input_data
    input_data = [x for x in input_data
                  if not any(x for y in excluded if str(y).lower() in str(x).lower())] if excluded else input_data
    return input_data


def convert_columns(input_data: pandas.DataFrame,
                    desired_type: str = 'str',
                    undesired_type: str = None,
                    included: list = None,
                    excluded: list = None):
    """
    Converts dataframe columns to desired type

    :param input_data: dataframe to convert
    :param desired_type: data type needed for the columns
    :param undesired_type: if specified convert only those that are detected as undesired type
    :param included: keywords by which to include columns
    :param excluded: keywords by which exclude columns
    :return: updated dataframe
    """
    if included is None and excluded is None:
        return input_data
    cols = filter_list(input_data=input_data.columns.to_list(), included=included, excluded=excluded)
    types = (input_data.dtypes.to_frame(name='Type').reset_index()
             .merge(pandas.DataFrame(data=cols, columns=['KEY']), on='KEY'))
    if undesired_type is not None:
        types = types[types['Type'] == undesired_type]
    to_change = types['KEY'].unique()
    input_data[to_change] = input_data[to_change].astype(desired_type)
    return input_data


def get_parent_id_from_triplets(input_data: pandas.DataFrame):
    """
    Gets parent id from triples data

    :param input_data: triplets data
    :return: dataframe containing parent ids
    """
    parent_id = (input_data[['ID']].rename(columns={'ID': 'PARENT_ID'})
                 .merge(input_data[['PARENT_ID']], on='PARENT_ID', how='right', indicator=True))
    return parent_id[parent_id['_merge'] == 'right_only'].drop_duplicates(keep='last')[['PARENT_ID']]


def parse_xml_to_dataframe(input_data, prefix_delimiter: str = '.', is_path: bool = False):
    """
    Parses xml to "Point" tableview

    :param is_path: if string and if it is path
    :param prefix_delimiter:
    :param input_data: xml document
    :return: pandas.DataFrame of point data, document type
    """
    triplets_data = parse_time_series(time_series_path=input_data, prefix_delimiter =prefix_delimiter, is_path=is_path)
    output = parse_triplets_to_point_tableview(triplets_data)
    parent_id = get_parent_id_from_triplets(input_data=triplets_data)
    document_type = triplets_data.merge(parent_id, on='PARENT_ID')
    document_type = document_type[document_type['KEY'] == 'Type']['VALUE'].unique().tolist()
    if len(document_type) == 0:
        logger.warning(f"Document type not detected")
    if len(document_type) > 1:
        logger.warning(f"Multiple document types detected")
    return output, document_type


def parse_and_filter_xml_to_dataframe(input_data,
                                      filter_dict: dict = None,
                                      elastic_check_included_keywords: list = None,
                                      elastic_check_excluded_keywords: list = None,
                                      elastic_check_type: str = ELASTIC_CHECK_DESIRED_TYPE):
    """
    Converts input xml to dataframe

    :param elastic_check_type: change columns to this data type
    :param elastic_check_excluded_keywords: keywords for columns to change
    :param elastic_check_included_keywords: keywords for columns to exclude from change
    :param input_data: bytes or string
    :param filter_dict: use key value pairs to filter the types
    :return: list of dictionaries
    """
    elastic_check_included_keywords = elastic_check_included_keywords or ELASTIC_CHECK_TYPE_INCLUDED
    elastic_check_excluded_keywords = elastic_check_excluded_keywords or ELASTIC_CHECK_TYPE_EXCLUDED
    output, doc_type = parse_xml_to_dataframe(input_data=input_data)
    output = convert_columns(input_data=output,
                             desired_type=elastic_check_type,
                             included=elastic_check_included_keywords,
                             excluded=elastic_check_excluded_keywords)
    if filter_dict and isinstance(filter_dict, dict):
        output = filter_dataframe(input_data=output, key_value_pairs=filter_dict)
    return output, doc_type


def parse_xml_to_list_of_dicts(input_data,
                               filter_dict: dict = None,
                               elastic_check_included_keywords: list = None,
                               elastic_check_excluded_keywords: list = None,
                               elastic_check_type: str = ELASTIC_CHECK_DESIRED_TYPE) -> Any | None:
    """
    Converts input xml to list of dictionaries

    :param elastic_check_type: change columns to this data type
    :param elastic_check_excluded_keywords: keywords for columns to change
    :param elastic_check_included_keywords: keywords for columns to exclude from change
    :param input_data: bytes or string
    :param filter_dict: use key value pairs to filter the types
    :return: list of dictionaries
    """
    output, _ = parse_and_filter_xml_to_dataframe(input_data=input_data,
                                                  filter_dict=filter_dict,
                                                  elastic_check_included_keywords=elastic_check_included_keywords,
                                                  elastic_check_excluded_keywords=elastic_check_excluded_keywords,
                                                  elastic_check_type=elastic_check_type)
    if not output.empty:
        return parse_dataframe_to_nested_dict(input_dataframe=output)
    return None


def parse_xml_dataframe_to_calculation_result(input_data,
                                              bid_document_types: dict = None,
                                              atc_document_types: dict = None,
                                              task: ProcurementCalculationTask = None,
                                              **kwargs):
    """
    Main function to parse xml file to CalculationResult

    :param atc_document_types: specify mappings for atc data to CalculationResult
    :param bid_document_types: specify mappings for bid data to CalculationResult
    :param input_data: xml file or path to it
    :param task: use this to update values
    :return: List of CalculationResult
    """
    point_dataframe, document_type = parse_xml_to_dataframe(input_data=input_data)
    bid_document_types = bid_document_types or BID_XML_RESULT_TYPES
    atc_document_types = atc_document_types or ATC_XML_RESULT_TYPES
    bid_type_value = bid_document_types.get(document_type[0])
    atc_type_value = atc_document_types.get(document_type[0])
    received_results = []
    task = task or get_task_from_environment()
    if bid_type_value is not None:
        single_result = CalculationResultBid()
        if task is not None:
            single_result.init_from_task(task)
        single_result.dataframe_to_object_list(input_data=point_dataframe, column_mapping=bid_type_value)
        received_results.append(single_result)
    if atc_type_value is not None:
        single_result = CalculationResultATC()
        if task is not None:
            single_result.init_from_task(task)
        single_result.dataframe_to_object_list(input_data=point_dataframe, column_mapping=atc_type_value)
        received_results.append(single_result)
    return received_results


if __name__ == '__main__':

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parse_app_properties(globals(), config.paths.config.transparency_data)
    parse_app_properties(globals(), config.paths.config.procurement)
    paths = [
        r'../../resources/examples/reserveAllocationResult_10X1001A1001A39W_20250120_1.xml',
        r'../../resources/examples/reserveAllocationResult_10X1001A1001A39W_20250120_2.xml',
        r"../resources/examples/LT_LV_ATC.xml",
        r"../../resources/examples/LV_EE_ATC.xml"
    ]

    one_level_up = os.path.join(os.getcwd().split('py')[0], os.pardir)
    send_to_elastic = True
    save_to_local_storage = True

    index_name = 'procurement-points'
    address = PY_ELASTICSEARCH_HOST
    file_counter = 1
    for single_path in paths:
        xml_path = get_file_path_from_root_by_name(file_name=os.path.basename(single_path), root_folder=one_level_up)
        triplets_dataframe = parse_time_series(xml_path)
        points_data = parse_triplets_to_point_tableview(triplets_dataframe)
        nested_dict = parse_dataframe_to_nested_dict(points_data)

        if send_to_elastic:
            dicts_with_ids = generate_ids_from_dict(input_data=nested_dict, id_fields=PY_PROD_ATC_INDEX_KEYS)
            Elastic.send_to_elastic_bulk(json_message_list=dicts_with_ids, server=address, index=index_name)

        if save_to_local_storage:
            time_moment_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            file_name_1 = f"file_{file_counter}_{time_moment_now}.csv"
            path_local_storage = '../'
            if path_local_storage:
                check_and_create_the_folder_path(path_local_storage)
                file_name_1 = path_local_storage.removesuffix('/') + '/' + file_name_1.removeprefix('/')
                points_data.to_csv(file_name_1)

    print("Done")
