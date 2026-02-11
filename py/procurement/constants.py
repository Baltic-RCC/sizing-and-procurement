import logging

import config
from py.common.config_parser import parse_app_properties
from py.common.functions import load_string_to_list_of_float, convert_input, escape_empty_or_none, \
    str_to_bool, parse_to_type, check_dict_to_dataclass
from py.data_classes.enums import RoleType, EICCodeType, ProcurementCalculationType, parse_to_enum, FlowDirectionType, \
    ExceededPercentType, NegativeValuesHandler, ExceededEnumOperator
from py.data_classes.task_classes import (EICArea, MessageCommunicator, NCBPCorrectionKey, update_list_of_objects)

parse_app_properties(globals(), config.paths.config.procurement)

QUANTILES_INDEX_NAME = 'percentage_level'
DEFAULT_QUERY = {'match_all': {}}
ENTSOE_PIVOT_VALUE = ['quantity', 'Point.quantity']

ATC_COLUMN_NAMES = ['out_domain.mRID', 'in_domain.mRID']

RCC = MessageCommunicator(mRID='38X-BALTIC-RSC-H',
                          name='Baltic RCC',
                          market_role_type=RoleType.REGIONAL_SECURITY_COORDINATOR,
                          function=EICCodeType.RCC)

DEFAULT_TEST_RECEIVER_1 = MessageCommunicator(mRID='RECEIVER_MRID_NO',
                                              name='Test TSO_Receiver',
                                              market_role_type=RoleType.SYSTEM_OPERATOR,
                                              function=EICCodeType.TSO)
DEFAULT_TEST_RECEIVER_2 = MessageCommunicator(mRID='RECEIVER_MRID_2',
                                              name='TSO2_Receiver',
                                              market_role_type=RoleType.SYSTEM_OPERATOR,
                                              function=EICCodeType.TSO)
DEFAULT_TEST_RECEIVER_3 = MessageCommunicator(mRID='RECEIVER_MRID_3',
                                              name='TSO3_Receiver',
                                              market_role_type=RoleType.SYSTEM_OPERATOR,
                                              function=EICCodeType.TSO)

AST = MessageCommunicator(mRID='10X1001A1001B54W',
                          name='AST',
                          market_role_type=RoleType.SYSTEM_OPERATOR,
                          function=EICCodeType.TSO)
ELERING = MessageCommunicator(mRID='10X1001A1001A39W',
                              name='Elering',
                              market_role_type=RoleType.SYSTEM_OPERATOR,
                              function=EICCodeType.TSO,
                              output_resolution='PT1H'
                              )
LITGRID = MessageCommunicator(mRID='10X1001A1001A55Y',
                              name='LITGRID',
                              market_role_type=RoleType.SYSTEM_OPERATOR,
                              function=EICCodeType.TSO)
ESTONIA = EICArea(mRID='10Y1001A1001A39I',
                  name='Estonia',
                  area_code='EE',
                  party=ELERING,
                  function=EICCodeType.LFC_AREA)
LATVIA = EICArea(mRID='10YLV-1001A00074',
                name='Latvia',
                area_code='LV',
                party=AST,
                function=EICCodeType.LFC_AREA)
LITHUANIA = EICArea(mRID='10YLT-1001A0008Q',
                    name='Lithuania',
                    area_code='LT',
                    party=LITGRID,
                    function=EICCodeType.LFC_AREA)
BALTICS = EICArea(mRID='10Y1001A1001A94A',
                  name='Baltics',
                  function=EICCodeType.LFC_BLOCK)


CORRECTION_KEYS = [NCBPCorrectionKey(country='Estonia',
                                     direction=FlowDirectionType.UP,
                                     borders=[('EE', 'LV'), ('LV', 'LT')]),
                   NCBPCorrectionKey(country='Estonia',
                                     direction=FlowDirectionType.DOWN,
                                     borders=[('LT', 'LV'), ('LV', 'EE')]),
                   NCBPCorrectionKey(country='Latvia',
                                     direction=FlowDirectionType.UP,
                                     borders= [('LV', 'EE'), ('LV', 'LT')]),
                   NCBPCorrectionKey(country='Latvia',
                                     direction=FlowDirectionType.DOWN,
                                     borders=[('EE', 'LV'), ('LT', 'LV')]),
                    NCBPCorrectionKey(country='Lithuania',
                                     direction=FlowDirectionType.UP,
                                     borders= [('LT', 'LV'), ('LV', 'EE')]),
                   NCBPCorrectionKey(country='Lithuania',
                                     direction=FlowDirectionType.DOWN,
                                     borders=[('EE', 'LV'), ('LV', 'LT')])]

DEFAULT_SENDER = [RCC]
DEFAULT_RECEIVER = [ELERING, AST, LITGRID, DEFAULT_TEST_RECEIVER_1, DEFAULT_TEST_RECEIVER_2, DEFAULT_TEST_RECEIVER_3]
AVAILABLE_SENDERS = [*DEFAULT_RECEIVER, *DEFAULT_SENDER]

PY_RECEIVER = convert_input(RECEIVER)
PY_SENDER = convert_input(SENDER)
if isinstance(PY_SENDER, list):
    PY_SENDER = [check_dict_to_dataclass(MessageCommunicator, x) for x in PY_SENDER]
elif isinstance(PY_SENDER, dict):
    PY_SENDER = [check_dict_to_dataclass(MessageCommunicator, PY_SENDER)]
if isinstance(PY_RECEIVER, list):
    PY_RECEIVER = [check_dict_to_dataclass(MessageCommunicator, x) for x in PY_RECEIVER]
elif isinstance(PY_RECEIVER, dict):
    PY_RECEIVER = [check_dict_to_dataclass(MessageCommunicator, PY_RECEIVER)]

FINAL_SENDER = update_list_of_objects(old_list=DEFAULT_SENDER, new_list=PY_SENDER, primary_key='mRID')
FINAL_RECEIVER = update_list_of_objects(old_list=DEFAULT_RECEIVER, new_list=PY_RECEIVER, primary_key='mRID')

DEFAULT_AREA = [ESTONIA, LATVIA, LITHUANIA]
DEFAULT_BLOCK = [BALTICS]
logger = logging.getLogger(__name__)
PY_PROCUREMENT_STARTING_PERCENT = load_string_to_list_of_float(PROCUREMENT_STARTING_PERCENT)
PY_PROCUREMENT_STEP_SIZE = load_string_to_list_of_float(PROCUREMENT_STEP_SIZE)
PY_PROCUREMENT_SEND_SIZE = load_string_to_list_of_float(PROCUREMENT_SEND_SIZE)
PY_PROCUREMENT_ENDING_PERCENT = load_string_to_list_of_float(PROCUREMENT_ENDING_PERCENT)
PY_ATC_QUERY = convert_input(ATC_QUERY)
PY_BID_QUERY = convert_input(BID_QUERY)

PY_BID_XMLS = convert_input(NCPB_XML_TYPES)
PY_ATC_XMLS = convert_input(ATC_XML_TYPES)

PY_CALCULATION_TIME_ZONE = CALCULATION_TIME_ZONE

PY_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD = INPUT_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD

PY_NEGATIVE_VALUE_POLICY = parse_to_enum(NEGATIVE_VALUE_POLICY, NegativeValuesHandler, NegativeValuesHandler.DO_NOTHING)

PY_DATA_PERIOD_TIME_DELTA = INPUT_DATA_PERIOD_TIME_DELTA
PY_DATA_PERIOD_START_DATE = INPUT_DATA_PERIOD_START_DATE
PY_DATA_PERIOD_END_DATE = INPUT_DATA_PERIOD_END_DATE
PY_DATA_PERIOD_OFFSET = INPUT_DATA_PERIOD_OFFSET

PY_CALCULATION_START_TIME = escape_empty_or_none(CALCULATION_TIMESTAMP_START)
PY_CALCULATION_END_TIME = escape_empty_or_none(CALCULATION_TIMESTAMP_END)
PY_CALCULATION_STEPS = escape_empty_or_none(CALCULATION_STEPS)
if PY_CALCULATION_STEPS:
    PY_CALCULATION_STEPS = int(PY_CALCULATION_STEPS)



PY_PERIOD_VALID_TIME_SHIFT = INPUT_DATA_PERIOD_AND_VALID_PERIOD_TIME_DELTA

PY_VALID_PERIOD_TIME_DELTA = VALID_PERIOD_TIME_DELTA
PY_VALID_PERIOD_START_DATE = VALID_PERIOD_START_DATE
PY_VALID_PERIOD_END_DATE = VALID_PERIOD_END_DATE
PY_VALID_PERIOD_OFFSET = VALID_PERIOD_OFFSET

PY_CALCULATION_TYPE = parse_to_enum(CALCULATION_TYPE, ProcurementCalculationType, ProcurementCalculationType.ALL)
PY_EXCEEDED_OPERATOR = parse_to_enum(EXCEEDED_OPERATOR, ExceededEnumOperator, ExceededEnumOperator.CASCADE)
DEFAULT_TYPE = ""

PY_OUTPUT_ATC_CODES = convert_input(OUTPUT_ATC_CODES)
PY_OUTPUT_NCPB_CODES = convert_input(OUTPUT_NCPB_CODES)
PY_EXCEEDED_NCPB_CODES = convert_input(EXCEEDED_NCPB_CODES)
PY_EXCEEDED_ATC_CODES = convert_input(EXCEEDED_ATC_CODES)

PY_VERSION_NUMBER = parse_to_type(VERSION_NUMBER, int)

POWER_MEASUREMENT_UNIT = escape_empty_or_none(POWER_MEASUREMENT_UNIT) or DEFAULT_TYPE
PERCENTAGE_MEASUREMENT_UNIT = escape_empty_or_none(PERCENTAGE_MEASUREMENT_UNIT) or DEFAULT_TYPE
CURVE_TYPE = escape_empty_or_none(CURVE_TYPE) or DEFAULT_TYPE
ENERGY_PRODUCT = escape_empty_or_none(ENERGY_PRODUCT) or DEFAULT_TYPE
CODING_SCHEME = escape_empty_or_none(CODING_SCHEME) or DEFAULT_TYPE
STATUS_TYPE = escape_empty_or_none(STATUS_TYPE) or DEFAULT_TYPE
PY_TASK_DESCRIPTION = escape_empty_or_none(TASK_DESCRIPTION)
PY_PROCUREMENT_SENDER_RCC = convert_input(PROCUREMENT_SENDER_RCC)
PY_PROCUREMENT_RECEIVER_TSO = convert_input(PROCUREMENT_RECEIVERS_TSO)
PY_PROCUREMENT_LFC_AREA = convert_input(PROCUREMENT_LFC_AREA)
PY_PROCUREMENT_LFC_BLOCK = convert_input(PROCUREMENT_LFC_BLOCK)
PY_OUTPUT_SEND_TO_ELASTIC = str_to_bool(str(OUTPUT_SEND_TO_ELASTIC).lower())
PY_OUTPUT_SEND_XML_TO_MINIO = str_to_bool(str(OUTPUT_SEND_XML_TO_MINIO).lower())

PY_OUTPUT_SEND_XLSX_TO_MINIO = str_to_bool(str(OUTPUT_SEND_XLSX_TO_MINIO).lower())
PY_OUTPUT_SEND_XML_OUT = str_to_bool(str(OUTPUT_SEND_XML_OUT).lower())
PY_OUTPUT_SEND_XLSX_OUT = str_to_bool(str(OUTPUT_SEND_XLSX_OUT).lower())
PY_OUTPUT_SEND_WITH_EDX = str_to_bool(str(OUTPUT_WITH_EDX).lower())
PY_OUTPUT_SEND_WITH_RABBIT = str_to_bool(str(OUTPUT_WITH_RABBIT).lower())

PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE = str_to_bool(str(OUTPUT_SAVE_XML_TO_LOCAL_STORAGE).lower())
PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE = str_to_bool(str(OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE).lower())

PY_EXCEL_ADD_INPUT_DATA = str_to_bool(str(EXCEL_ADD_INPUT_DATA).lower())
PY_EXCEL_ADD_OBJECT_LIST = str_to_bool(str(EXCEL_ADD_OBJECT_LIST).lower())
PY_EXCEL_ADD_QUANTILE_DATA = str_to_bool(str(EXCEL_ADD_QUANTILE_DATA).lower())
PY_EXCEL_ADD_PIVOTED_DATA = str_to_bool(str(EXCEL_ADD_PIVOTED_DATA).lower())
PY_EXCEL_ADD_CORRECTED_NCPB = str_to_bool(str(EXCEL_ADD_CORRECTED_NCPB).lower())

PY_PROPOSED_RMQ_HEADERS = convert_input(PROPOSED_RMQ_HEADERS)
PY_REALISING_RMQ_HEADERS = convert_input(REALISING_RMQ_HEADERS)
PY_EXCEEDED_RMQ_HEADERS = convert_input(EXCEEDED_RMQ_HEADERS)

PY_TABLE_COUNTRY_KEY = TABLE_COUNTRY_KEY

PY_ATC_FILTER = convert_input(ATC_FILTER)
PY_NCPB_FILTER = convert_input(NCPB_FILTER)

PY_XML_PREFIX = convert_input(XML_PREFIX)
PY_XML_PREFIX = [PY_XML_PREFIX] if isinstance(PY_XML_PREFIX, str) else PY_XML_PREFIX

PY_EXCEEDED_PREFIX = convert_input(EXCEEDED_PREFIX)
PY_EXCEEDED_PREFIX = [PY_EXCEEDED_PREFIX] if isinstance(PY_EXCEEDED_PREFIX, str) else PY_EXCEEDED_PREFIX
PY_EXCEEDED_TYPE = parse_to_enum(EXCEEDED_TYPE, ProcurementCalculationType, ProcurementCalculationType.ALL)
PY_EXCEEDED_PERCENT = parse_to_enum(EXCEEDED_PERCENT, ExceededPercentType, ExceededPercentType.MAX)
PY_EXCEEDED_INITIAL = str_to_bool(str(EXCEEDED_INITIAL).lower())
PY_EXCEEDED_REPORT = str_to_bool(str(EXCEEDED_REPORT).lower())
PY_EXCEEDED_SIGN = FOR_SKAIDRITE_TO_DECIDE

PY_ATC_TIME_KEY_FROM = escape_empty_or_none(ATC_TIME_KEY_FROM) or 'valid_from'
PY_ATC_TIME_KEY_TO = escape_empty_or_none(ATC_TIME_KEY_TO) or 'valid_to'
PY_NCPB_TIME_KEY_FROM = escape_empty_or_none(NCPB_TIME_KEY_FROM) or 'valid_from'
PY_NCPB_TIME_KEY_TO = escape_empty_or_none(NCPB_TIME_KEY_TO) or 'valid_to'
PY_NCPB_FILL_NA = escape_empty_or_none(NCPB_FILL_NA)
ENTSOE_INDEX_COLUMNS = [PY_ATC_TIME_KEY_FROM, PY_ATC_TIME_KEY_TO]

CALCULATED_COUNTRIES = PY_PROCUREMENT_LFC_AREA or ['Estonia', 'Latvia', 'Lithuania']

XML_FOLDER_TO_STORE = r"E:\margus.ratsep\sizing_of_reserves\xmls"
EXCEL_FOLDER_TO_STORE = r"E:\margus.ratsep\sizing_of_reserves\reports"

METADATA_KEY = 'meta_data'
CORRECTED_NCPB_KEY = 'Corrected'
