import logging

import config
from py.common.config_parser import parse_app_properties
from py.common.functions import escape_empty_or_none, load_string_to_list_of_float, parse_to_type, convert_input, \
    str_to_bool
from py.data_classes.enums import parse_to_enum, OutputFileType, ProcurementCalculationType, NameValueOfEnum
from py.procurement.constants import FINAL_SENDER, PY_PROCUREMENT_RECEIVER_TSO, FINAL_RECEIVER, \
    PY_PROCUREMENT_SENDER_RCC
from py.handlers.elastic_handler import PY_PROCUREMENT_PROPOSED_INDEX, PY_PROCUREMENT_REALISED_INDEX, \
    PY_SIZING_AND_PROCUREMENT_LOGS_INDEX

logger = logging.getLogger(__name__)

parse_app_properties(globals(), config.paths.config.json_to_minio)


PY_DOWNLOAD_TYPE = parse_to_enum(OUTPUT_DOWNLOAD_TYPE, ProcurementCalculationType, ProcurementCalculationType.ALL)

PY_TIME_FROM = escape_empty_or_none(OUTPUT_TIME_FROM)
PY_TIME_TO = escape_empty_or_none(OUTPUT_TIME_TO)
PY_TIME_OFFSET = escape_empty_or_none(OUTPUT_TIME_OFFSET)
PY_TIME_DELTA = escape_empty_or_none(OUTPUT_TIME_DELTA)
PY_OUTPUT_TIME_KEY = convert_input(escape_empty_or_none(OUTPUT_TIME_KEY), False)
PY_OUTPUT_EXCEL_COLUMNS = convert_input(escape_empty_or_none(OUTPUT_EXCEL_COLUMNS))
# PY_POINT_RESOLUTION = escape_empty_or_none(OUTPUT_POINT_RESOLUTION) or 'P1D'
PY_POINT_RESOLUTION = escape_empty_or_none(OUTPUT_POINT_RESOLUTION)

class ElasticIndexType(NameValueOfEnum):
    PROPOSED_INDEX = PY_PROCUREMENT_PROPOSED_INDEX
    REALISED_INDEX = PY_PROCUREMENT_REALISED_INDEX
    LOGS_INDEX = PY_SIZING_AND_PROCUREMENT_LOGS_INDEX
PY_OUTPUT_DATA_ELASTIC_INDEX = OUTPUT_DATA_ELASTIC_INDEX
PY_CUSTOM_OUTPUT_DATA_ELASTIC_INDEX = escape_empty_or_none(CUSTOM_OUTPUT_DATA_ELASTIC_INDEX)

PY_ELASTIC_INDEX = parse_to_enum(OUTPUT_DATA_ELASTIC_INDEX, ElasticIndexType, ElasticIndexType.PROPOSED_INDEX)
if PY_CUSTOM_OUTPUT_DATA_ELASTIC_INDEX is None:
    PY_ELASTIC_TYPE = PY_ELASTIC_INDEX
    PY_ELASTIC_INDEX = PY_ELASTIC_INDEX.value
else:
    PY_ELASTIC_TYPE = None
    PY_ELASTIC_INDEX = PY_CUSTOM_OUTPUT_DATA_ELASTIC_INDEX

PY_START_X = load_string_to_list_of_float(OUTPUT_QUANTILE_START)
PY_STEP_Y = load_string_to_list_of_float(OUTPUT_QUANTILE_STEP)
PY_STOP_Z = load_string_to_list_of_float(OUTPUT_QUANTILE_STOP)

PY_OUTPUT_VERSION_NUMBER = parse_to_type(OUTPUT_VERSION_NUMBER, int)
PY_OUTPUT_FILE_TYPE = parse_to_enum(OUTPUT_FILE_TYPE, OutputFileType, OutputFileType.XML)
PY_OUTPUT_CUSTOM_QUERY = convert_input(OUTPUT_CUSTOM_QUERY)

PY_OUTPUT_FILE_TO_MINIO = str_to_bool(str(OUTPUT_FILE_TO_MINIO).lower())
PY_OUTPUT_FILE_TO_RABBIT = str_to_bool(str(OUTPUT_FILE_TO_RABBIT).lower())

xml_sender = next(iter(sender for sender in FINAL_SENDER if sender.value_of(PY_PROCUREMENT_SENDER_RCC)))
xml_receivers = [receiver for receiver_name in PY_PROCUREMENT_RECEIVER_TSO
                 for receiver in FINAL_RECEIVER if receiver.value_of(receiver_name)]

realised_prefixes = ['TSO-Realised']
proposed_prefixes = ['RCC-Proposed']
prefixes_to_use = realised_prefixes if PY_ELASTIC_TYPE == ElasticIndexType.REALISED_INDEX else proposed_prefixes

