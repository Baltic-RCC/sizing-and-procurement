import copy
import logging
import os
from datetime import datetime

import pandas
from lxml import etree

from py.common.ref_constants import QUANTILE_STOP_KEY, QUANTILE_STEP_KEY, QUANTILE_START_KEY, POINT_RESOLUTION_KEY, \
    RESULT_TYPE_KEY, VERSION_KEY
from py.data_classes.elastic.elastic_data_models import DataToXMLPoint
from py.data_classes.task_classes import MessageCommunicator, TIME_KEYS
from py.data_classes.xml.balancing_market_document import BalancingXMLDocument, BalancingXMLTimeSeries, \
    BALANCING_XML_DOCUMENT_NAME, BALANCING_XML_TO_RESULT_MAP
from py.data_classes.xml.capacity_document import CapacityXMLDocument, CapacityXMLTimeSeries, CAPACITY_DOCUMENT_NAME, \
    CAPACITY_XML_TO_RESULT_MAP
from py.data_classes.xml.merit_order_list_document import MeritOrderXMLTimeSeries, MeritOrderXMLDocument, \
    MERIT_ORDER_DOCUMENT_NAME, MERIT_XML_TO_RESULT_MAP
from py.data_classes.xml.reserve_bid_document import ReserveBidXMLTimeSeries, ReserveBidXMLDocument, \
    RESERVE_XML_TO_RESULT_MAP, RESERVE_BID_DOCUMENT_NAME

from py.handlers.elastic_handler import get_data_from_elastic_by_time, PY_PROCUREMENT_PROPOSED_INDEX, merge_queries
from py.procurement.constants import PY_BID_XMLS, PY_ATC_XMLS, PY_PROCUREMENT_STARTING_PERCENT, PY_PROCUREMENT_ENDING_PERCENT, \
    PY_PROCUREMENT_STEP_SIZE
from py.data_classes.enums import ProcurementCalculationType, ValueOfEnum, NameValueOfEnum
from py.common.functions import get_file_path_from_root_by_name
from py.common.time_functions import parse_duration, str_to_datetime, time_delta_to_str, \
    get_datetime_columns_of_data_frame, set_timezone

logger = logging.getLogger(__name__)

capacity_schema = r'../../resources/schemas/iec62325-451-3-capacity_v8_3.xsd'
reserve_schema = r'../../resources/schemas/iec62325-451-7-reservebiddocument_v7_6.xsd'
balancing_schema = r'../../resources/schemas/iec62325-451-6-balancing_v4_5.xsd'
merit_schema = r'../../resources/schemas/iec62325-451-7-moldocument_v7_3.xsd'

level_up = os.path.join(os.getcwd().split('py')[0], os.pardir)
capacity_path = get_file_path_from_root_by_name(file_name=os.path.basename(capacity_schema), root_folder=level_up)
reserve_path = get_file_path_from_root_by_name(file_name=os.path.basename(reserve_schema), root_folder=level_up)
balancing_path = get_file_path_from_root_by_name(file_name=os.path.basename(balancing_schema), root_folder=level_up)
merit_path = get_file_path_from_root_by_name(file_name=os.path.basename(merit_schema), root_folder=level_up)

DEFAULT_POINT_RESOLUTION = 'P1D'


def get_element_or_first_from_list(input_data):
    if isinstance(input_data, list):
        return input_data[0]
    if input_data:
        return input_data
    return None


def get_version_number_from_elastic(calculation_type: ProcurementCalculationType = None,
                                    point_resolution: str = None,
                                    quantile_start_value=PY_PROCUREMENT_STARTING_PERCENT,
                                    quantile_step_value: float | str = PY_PROCUREMENT_STEP_SIZE,
                                    quantile_stop_value: float | str = PY_PROCUREMENT_ENDING_PERCENT,
                                    valid_from: str | datetime = None,
                                    valid_to: str | datetime = None,
                                    time_keys: list | str = None,
                                    additional_elements: dict = None,
                                    results_index: str = PY_PROCUREMENT_PROPOSED_INDEX):
    """

    :param calculation_type:
    :param point_resolution:
    :param quantile_start_value:
    :param quantile_step_value:
    :param quantile_stop_value:
    :param valid_from:
    :param valid_to:
    :param time_keys:
    :param additional_elements:
    :param results_index:
    :return:
    """
    components = []
    time_keys = time_keys or TIME_KEYS
    version_key = VERSION_KEY
    if calculation_type:
        c_val = calculation_type.value if isinstance(calculation_type, ProcurementCalculationType) else calculation_type
        components.append({RESULT_TYPE_KEY: str(c_val)})
    if point_resolution:
        components.append({POINT_RESOLUTION_KEY: time_delta_to_str(point_resolution)})
    if quantile_start_value := get_element_or_first_from_list(quantile_start_value):
        components.append({QUANTILE_START_KEY: quantile_start_value})
    if quantile_step_value := get_element_or_first_from_list(quantile_step_value):
        components.append({QUANTILE_STEP_KEY: quantile_step_value})
    if quantile_stop_value := get_element_or_first_from_list(quantile_stop_value):
        components.append({QUANTILE_STOP_KEY: quantile_stop_value})
    data_query = {"bool": {"must": [{'match': component} for component in components]}}
    if additional_elements and isinstance(additional_elements, dict):
        data_query = merge_queries(query_dict=data_query, merge_dict=additional_elements)
    data_dataframe = get_data_from_elastic_by_time(start_time_value=valid_from,
                                                   end_time_value=valid_to,
                                                   elastic_query=data_query,
                                                   elastic_index=results_index,
                                                   dict_to_flat=False,
                                                   time_interval_key=time_keys)
    if data_dataframe.empty:
        logger.info(f"No data about versions with these parameters found")
        return None
    data_dataframe[version_key] = pandas.to_numeric(data_dataframe[version_key], downcast='integer', errors='coerce')
    latest_version = data_dataframe[version_key].max()
    logger.info(f"Latest version for these parameters is {latest_version}")
    return latest_version


class DocumentXML:
    """
    Additional class to house data related to the xmls

    :param document_name: Specify name of the class instance that is xml document
    :param series_name: Specify name of the class instance that will be Timeseries within the XML document
    :param xsd_schema_path:  Specify the path to the schemas
    :param file_name_prefix: Specify prefix for the names (good to distinguish them)
    """

    def __init__(self, document_name, series_name, xsd_schema_path: str, file_name_prefix: str):
        """
        Constructor
        """
        self.document_name = document_name
        self.series_name = series_name
        self.xsd_schema_path = xsd_schema_path
        self.file_name_prefix = file_name_prefix


class XMLType(ValueOfEnum):
    """
    XML output types so far
    """
    BALANCING_DOCUMENT = DocumentXML(document_name=BalancingXMLDocument,
                                     series_name=BalancingXMLTimeSeries,
                                     xsd_schema_path=balancing_path,
                                     file_name_prefix='Balancing')
    MERIT_ORDER_DOCUMENT = DocumentXML(document_name=MeritOrderXMLDocument,
                                       series_name=MeritOrderXMLTimeSeries,
                                       xsd_schema_path=merit_path,
                                       file_name_prefix='Merit_Order')
    RESERVE_BID_DOCUMENT = DocumentXML(document_name=ReserveBidXMLDocument,
                                       series_name=ReserveBidXMLTimeSeries,
                                       xsd_schema_path=reserve_path,
                                       file_name_prefix='Reserve_Bid')
    CAPACITY_DOCUMENT = DocumentXML(document_name=CapacityXMLDocument,
                                    series_name=CapacityXMLTimeSeries,
                                    xsd_schema_path=capacity_path,
                                    file_name_prefix='Capacity')


class XMLToResultType(NameValueOfEnum):
    BALANCING_DOCUMENT = {BALANCING_XML_DOCUMENT_NAME: BALANCING_XML_TO_RESULT_MAP}
    MERIT_ORDER_DOCUMENT = {MERIT_ORDER_DOCUMENT_NAME: MERIT_XML_TO_RESULT_MAP}
    RESERVE_BID_DOCUMENT = {RESERVE_BID_DOCUMENT_NAME: RESERVE_XML_TO_RESULT_MAP}
    CAPACITY_DOCUMENT = {CAPACITY_DOCUMENT_NAME: CAPACITY_XML_TO_RESULT_MAP}


BID_XML_DOCUMENT_TYPES = [XMLType.value_of(x) for x in PY_BID_XMLS]
ATC_XML_DOCUMENT_TYPES = [XMLType.value_of(x) for x in PY_ATC_XMLS]

BID_XML_RESULT_TYPES = {y_key: y_value for y in [XMLToResultType.value_of(x) for x in PY_BID_XMLS]
                        for y_key, y_value in y.value.items() if isinstance(y.value, dict)}
ATC_XML_RESULT_TYPES = {y_key: y_value for y in [XMLToResultType.value_of(x) for x in PY_ATC_XMLS]
                        for y_key, y_value in y.value.items() if isinstance(y.value, dict)}


def validate_xml(xml_string, xsd_file):
    """
    Function to validate xml strings

    :param xml_string: input xml
    :param xsd_file: loaded schema
    :return: True if ok, False otherwise
    """
    try:
        schema = etree.XMLSchema(file=xsd_file)

        xml_doc = etree.fromstring(xml_string)

        schema.assertValid(xml_doc)
        logger.debug(f"XML validation successful for {xml_string[:150]}")
        return True

    except etree.XMLSyntaxError as e:
        logger.error(f"In {xml_string[:150]} XML syntax error found: {e}")
        return False
    except etree.DocumentInvalid as e:
        logger.error(f"Document {xml_string[:150]} is invalid: {e}")
        return False


def get_first_from_column(input_data: pandas.DataFrame, column_name: str, strict: bool = True):
    """
    Gets first value from dataframe column

    :param input_data: input dataframe
    :param column_name: name of column
    :param strict: true then returns if unique values is 1
    :return: value from column
    """
    column_unique = input_data[column_name].unique().tolist()
    if (strict and len(column_unique) == 1) or (not strict and len(column_unique) > 1):
        return next(iter(column_unique))
    return None


def get_value_from_column(input_data: pandas.DataFrame, column_string: str, strict: bool = True):
    """
    Gets first value from dataframe column found by column_string

    :param input_data: input dataframe
    :param column_string: string to search from column
    :param strict: true then returns if unique values is 1
    :return: value if found
    """
    new_unit = None
    column_names = [column_name
                    for column_name in input_data.columns.to_list()
                    if column_string.lower() in str(column_name).lower()]
    if len(column_names) > 1:
        column_names = [column_name
                        for column_name in input_data.columns.to_list()
                        if column_string.lower() == str(column_name).lower()]
    if len(column_names) == 1:
        new_unit = get_first_from_column(input_data=input_data, column_name=column_names[0], strict=strict)
    return new_unit


def check_resolutions(receiver: MessageCommunicator, point_resolution: str, values: list[DataToXMLPoint]):
    """
    If receiver specifies an output resolution then this
    1) Checks if the receiver resolution is smaller than one from data, if not then returns data resolution
    2) If first is smaller calculates the scale. if scale is less than (integer division) then returns existing
    3) Interpolates the data to receiver resolution

    :param receiver: receiver instance
    :param point_resolution: data resolution
    :param values: data
    :return: updated resolution, updated data
    """
    if not receiver.output_resolution:
        return point_resolution, values
    output_resolution = parse_duration(receiver.output_resolution)
    input_resolution = parse_duration(point_resolution)
    if input_resolution <= output_resolution:
        return point_resolution, values
    new_scale = int(input_resolution / output_resolution)
    if new_scale < 2:
        return point_resolution, values
    new_values = []
    new_counter = 1
    for j in range(len(values)):
        old_value = values[j]
        for i in range(int(new_scale)):
            new_values.append(DataToXMLPoint(position=new_counter,
                                             primary_quantity=old_value.primary_quantity,
                                             secondary_quantity=old_value.secondary_quantity))
            new_counter = new_counter + 1
    return output_resolution, new_values


def handle_dataframe_timezone_excel(input_data: pandas.DataFrame,
                                    columns: list = None,
                                    timezone: str = 'UTC',
                                    unlocalize: bool = True):
    """
    As Excel does not like date-time objects that contain timezone information then this escapes it

    :param unlocalize: escape timezone, set this true if send it to excel
    :param input_data:input dataframe
    :param columns: columns to be escaped, if not provided then datetime columns will be taken
    :param timezone: Convert to some timezone before removing it
    :return: updated dataframe
    """
    if columns is None:
        columns = get_datetime_columns_of_data_frame(input_data=input_data)
    if not isinstance(columns, list):
        columns = [columns]
    updated_dataframe = copy.deepcopy(input_data)
    updated_dataframe = str_to_datetime(updated_dataframe, columns)
    for column in columns:
        updated_dataframe[column] = updated_dataframe[column].apply(lambda x: set_timezone(x, time_zone=timezone))
        if updated_dataframe[column].dt.tz is None:
            updated_dataframe[column] = updated_dataframe[column].dt.tz_localize(tz=timezone)
        updated_dataframe[column] = updated_dataframe[column].dt.tz_convert(tz=timezone)
        if unlocalize:
            updated_dataframe[column] = updated_dataframe[column].dt.tz_localize(None)
    return updated_dataframe
