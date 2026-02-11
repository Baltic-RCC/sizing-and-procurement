import datetime
import itertools
import logging
import os
import sys
from enum import Enum

import pandas
from pandas import Index

from py.common.df_functions import filter_dataframe
from py.common.functions import calculate_start_and_end_date, get_random_string_length
from py.common.time_functions import str_to_datetime
from py.data_classes.results.calculation_result_main import get_senders_from_dataframe
from py.handlers.elastic_handler import get_data_from_elastic_by_time, dict_to_and_or_query, merge_queries, \
    get_data_from_elastic
from py.data_classes.enums import ProcurementCalculationType, MessageType, ProcessType, OutputFileType
from py.data_classes.task_classes import MessageCommunicator, TIME_KEYS, Domain, ProcurementCalculationTask
from py.handlers.rabbit_handler import PY_RMQ_EXCHANGE
from py.parsers.parser_constants import PY_DOWNLOAD_TYPE, PY_TIME_TO, PY_TIME_FROM, PY_POINT_RESOLUTION, PY_START_X, \
    PY_STEP_Y, PY_STOP_Z, PY_OUTPUT_VERSION_NUMBER, PY_OUTPUT_CUSTOM_QUERY, PY_ELASTIC_INDEX, PY_OUTPUT_FILE_TYPE, \
    xml_receivers, xml_sender, prefixes_to_use, PY_OUTPUT_FILE_TO_MINIO, PY_OUTPUT_FILE_TO_RABBIT, PY_ELASTIC_TYPE, \
    PY_OUTPUT_TIME_KEY, PY_OUTPUT_EXCEL_COLUMNS
from py.procurement.constants import (PY_VALID_PERIOD_OFFSET, PY_CALCULATION_TIME_ZONE, PY_VALID_PERIOD_TIME_DELTA,
                                      XML_FOLDER_TO_STORE, EXCEL_FOLDER_TO_STORE, PY_PROPOSED_RMQ_HEADERS,
                                      PY_EXCEL_ADD_CORRECTED_NCPB)
from py.procurement.procurement_output import generate_excel_from_dataframes, handle_parsed_output
from py.common.ref_constants import PROCESS_TYPE_KEY, VERSION_KEY, MESSAGE_TYPE_KEY, LFC_BLOCK_MRID_KEY, SENDER_MRID_KEY, \
    QUANTILE_STOP_KEY, QUANTILE_STEP_KEY, QUANTILE_START_KEY, POINT_RESOLUTION_KEY, RESULT_TYPE_KEY
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_bid import CalculationResultBid

logger = logging.getLogger(__name__)

UNIQUE_COLUMN_KEYWORDS = ['timestamp', 'version', 'available.value', 'calculation_date',
                          'mrid', 'index', '@timestamp', 'sender.mRID', 'description']


def convert_to_calculation_result(input_data: pandas.DataFrame,
                                  download_type: ProcurementCalculationType,
                                  valid_from: str | datetime.datetime = None,
                                  valid_to: str | datetime.datetime = None,
                                  message_type: str | MessageType = None,
                                  process_type: str | ProcessType = None,
                                  task: ProcurementCalculationTask = None,
                                  sender: MessageCommunicator = None,
                                  receivers: list[MessageCommunicator] = None):
    """
    Processes input dataframe to calculation result

    :param process_type: Overwrite process type
    :param message_type: Overwrite message type
    :param valid_to: Overwrite valid_to
    :param valid_from: Overwrite valid_from
    :param task: if given update default fields
    :param input_data: input dataframe
    :param download_type: Download type (ATC or NCPB)
    :param sender: MessageCommunicator instance as a sender necessary for creating xmls
    :param receivers: list of MessageCommunicator instances as receivers necessary for creating xmls
    :return: list of calculation results
    """
    input_data = str_to_datetime(data=input_data, columns=TIME_KEYS)
    message_types = [MessageType.value_of(x) for x in input_data[MESSAGE_TYPE_KEY].unique().tolist()]
    process_types = [ProcessType.value_of(x) for x in input_data[PROCESS_TYPE_KEY].unique().tolist()]
    senders = get_senders_from_dataframe(input_data=input_data, sender_column_name=SENDER_MRID_KEY)
    senders = sender if senders is None else senders
    version_numbers = input_data[VERSION_KEY].unique().tolist()
    lf_block_ids = input_data[LFC_BLOCK_MRID_KEY].unique().tolist()

    f_pairs = {MESSAGE_TYPE_KEY: message_types,
               PROCESS_TYPE_KEY: process_types,
               SENDER_MRID_KEY: senders,
               VERSION_KEY: version_numbers,
               LFC_BLOCK_MRID_KEY: lf_block_ids}
    f_pairs = {k: v for k, v in f_pairs.items() if (v is not None or (isinstance(v, list) and len(v) == 0))}
    combinations = [dict(zip(f_pairs.keys(), values)) for values in itertools.product(*f_pairs.values())]

    calculation_results = []
    for combo in combinations:
        filter_values = {k:v.value if isinstance(v, Enum) else v for k,v in combo.items()}
        filter_values = {k: v.mRID if isinstance(v, Domain) else v for k, v in filter_values.items()}
        df_slice = filter_dataframe(input_data=input_data, key_pairs=filter_values)
        if not df_slice.empty:
            logger.info(f"Processing {','.join([':'.join([str(x), str(y)]) for x, y in filter_values.items()])}")
            if download_type == ProcurementCalculationType.ATC:
                calculation_result = CalculationResultATC(calculation_type=download_type,
                                                          input_data=input_data,
                                                          process_type=combo.get(PROCESS_TYPE_KEY),
                                                          message_type=combo.get(MESSAGE_TYPE_KEY),
                                                          sender=combo.get(SENDER_MRID_KEY),
                                                          version=combo.get(VERSION_KEY),
                                                          lfc_block=combo.get(LFC_BLOCK_MRID_KEY),
                                                          receivers=receivers)
            elif download_type == ProcurementCalculationType.NCPB:
                calculation_result = CalculationResultBid(calculation_type=download_type,
                                                          input_data=input_data,
                                                          process_type=combo.get(PROCESS_TYPE_KEY),
                                                          message_type=combo.get(MESSAGE_TYPE_KEY),
                                                          sender=combo.get(SENDER_MRID_KEY),
                                                          version=combo.get(VERSION_KEY),
                                                          lfc_block=combo.get(LFC_BLOCK_MRID_KEY),
                                                          receivers=receivers)
            else:
                continue
            calculation_result.init_from_task(task=task, overwrite=False)
            calculation_result.dataframe_to_object_list(input_data=df_slice,
                                                        valid_from=valid_from,
                                                        valid_to=valid_to,
                                                        message_type=message_type,
                                                        process_type=process_type)
            calculation_results.append(calculation_result)
    return calculation_results


def get_matches_from_multilevel_index(input_index: Index, search_col: str | list | tuple):
    """
    Matches index names based on the keywords

    :param input_index: pandas index (or columns)
    :param search_col: keyword(s) to search
    :return: found matches or empty list
    """
    search_col = [search_col] if isinstance(search_col, str) else search_col
    matches =[]
    if 1 <= len(search_col) <= input_index.nlevels:
        for single_item in input_index.to_list():
            single_list = [single_item] if isinstance(single_item, str) else single_item
            if all(x in single_list for x in search_col):
                matches.append(single_item)
    return matches


def delete_columns(input_dataframe: pandas.DataFrame, columns_delete: list | str = None, delete_empty: bool = True):
    """
    Cleans up the dataframe (deletes columns not needed or containing empty values. NB! both bid and ATC are using
    the same index for storing the values)

    :param input_dataframe: input dataframe
    :param columns_delete: specify columns needed to be deleted explicitly
    :param delete_empty: If true deletes columns that only contain empty values
    :return: cleaned dataframe
    """
    if isinstance(columns_delete, str):
        columns_delete = [columns_delete]
    if not columns_delete:
        columns_delete = []
    matches = []
    for col_name in columns_delete:
        matches.extend(get_matches_from_multilevel_index(input_index=input_dataframe.columns, search_col=col_name))
    columns_delete = matches
    if delete_empty:
        for column in input_dataframe.columns.to_list():
            if input_dataframe[column].isnull().all():
                columns_delete.append(column)
    if columns_delete:
        return input_dataframe.drop(columns=columns_delete)
    return input_dataframe


def get_max_from_columns(input_data, name_key):
    """
    Filters dataframe to max value by the column names indicated (NB! keep in mind datatypes)

    :param input_data: input dataframe
    :param name_key: list of columns
    :return: filtered dataframe
    """
    return input_data[input_data[name_key] == max(input_data[name_key])].reset_index()


def filter_dataframe_to_latest_by_key(input_dataframe: pandas.DataFrame,
                                      variable_columns: list = None,
                                      fixed_columns: list = None,
                                      key_column: str = 'version'):
    """
    Filters dataframe to the latest version by index columns

    :param input_dataframe: input dataframe
    :param variable_columns: specify columns that can change for groups (if fixed columns is none)
    :param fixed_columns: set this if grouping by specific keys is needed (overwrites variable_columns)
    :param key_column: name of the version column
    :return: filtered dataframe
    """
    if input_dataframe.empty:
        return input_dataframe
    input_columns = input_dataframe.columns.to_list()
    key_column = next(iter([column for column in input_columns if key_column in column.lower()]), None)
    if fixed_columns is None:
        variable_columns = variable_columns or UNIQUE_COLUMN_KEYWORDS
        fixed_columns = [column for column in input_columns if
                         not any(key_word.lower() in column.lower() for key_word in variable_columns)]
    new_key = key_column + get_random_string_length()
    input_dataframe[new_key] = pandas.to_numeric(input_dataframe[key_column], downcast='integer', errors='coerce')
    max_versions = input_dataframe.groupby(fixed_columns, dropna=False)[new_key].transform('max')
    output =input_dataframe[input_dataframe[new_key] == max_versions]
    output = output.drop(columns=[new_key])
    return output


def is_pycharm():
    return os.getenv("PYCHARM_HOSTED") is not None


def handle_json_output(results: list,
                         prefixes: list,
                         output_local: bool = False,
                         excel_columns: list = None,
                         output_minio: bool = PY_OUTPUT_FILE_TO_MINIO,
                         output_rabbit: bool = PY_OUTPUT_FILE_TO_RABBIT,
                         output_file_type: OutputFileType = OutputFileType.XML):
    """
    For sending out the output

    :param results: CalculationResult type instances
    :param prefixes: prefixes for file names
    :param excel_columns: indicate excel columns to keep in Excel sheets
    :param output_local: save to local storage (guarded by pycharm)
    :param output_minio: save output to minio
    :param output_rabbit: send results to rabbit
    :param output_file_type: output file type
    :return: None
    """
    xml_minio = (output_file_type == OutputFileType.BOTH or output_file_type == OutputFileType.XML) and output_minio
    xml_rabbit = (output_file_type == OutputFileType.BOTH or output_file_type == OutputFileType.XML) and output_rabbit
    xlsx_minio = (output_file_type == OutputFileType.BOTH or output_file_type == OutputFileType.XLSX) and output_minio
    xlsx_rabbit = (output_file_type == OutputFileType.BOTH or output_file_type == OutputFileType.XLSX) and output_rabbit
    handle_parsed_output(results=results,
                         output_to_elastic=False,
                         elastic_index=None,
                         prefixes=prefixes,
                         excel_columns=excel_columns,
                         output_xml_minio=xml_minio,
                         output_xml_rabbit=xml_rabbit,
                         output_xml_local=output_local,
                         rabbit_exchange=PY_RMQ_EXCHANGE,
                         rabbit_headers=PY_PROPOSED_RMQ_HEADERS,
                         xml_local_path=XML_FOLDER_TO_STORE,
                         output_xlsx_minio=xlsx_minio,
                         output_xlsx_rabbit=xlsx_rabbit,
                         output_xlsx_local=output_local,
                         add_corrected_ncpb=PY_EXCEL_ADD_CORRECTED_NCPB,
                         xlsx_local_path=EXCEL_FOLDER_TO_STORE)


def get_calculation_results_from_elastic(start_date_time: str | datetime.datetime = None,
                                         end_date_time: str | datetime.datetime = None,
                                         offset_value: str | datetime.timedelta = None,
                                         time_delta_value: str | datetime.timedelta = None,
                                         time_keys: list | str = None,
                                         time_zone: str = None,
                                         elastic_index: str = None,
                                         download_type: ProcurementCalculationType = None,
                                         point_resolution: str | datetime.timedelta = None,
                                         start_x: str | int = None,
                                         step_y: str | int = None,
                                         stop_z: str | int = None,
                                         version_number: str | int = None,
                                         custom_query = None):
    """
    Queries proposed realised data from elastic

    :param elastic_index: Specify the index from where to download the data. Calculated results is default
    :param start_date_time: Specify the start time, P1D from current day will default
    :param end_date_time: Specify the end time, P2D from current day will default
    :param offset_value: if no start time specify offset from now (currently it is P1D)
    :param time_delta_value: if no end time specified
    :param time_keys: time keys to be used for querying. Default is valid_from, valid_to
    :param time_zone: time zone if needed
    :param download_type: download type: NCPB, ATC or ALL, All will be default
    :param point_resolution: points resolution, P1D will be used as default
    :param start_x: X value 90 will be used as default
    :param step_y:  Y value 0.1 will be used as default
    :param stop_z: Stop value, 100 will be used as default
    :param version_number: Version number, latest will be used as default
    :param custom_query: if needed something like {'message_type': 'A26', 'process_type': 'A14', 'business_type': 'C01'}
    :return: received data
    """
    time_keys = TIME_KEYS if time_keys is None else time_keys
    offset_value = offset_value if not start_date_time else None
    time_delta_value = time_delta_value if not end_date_time else None

    end_time, start_time = calculate_start_and_end_date(start_date_time=end_date_time,
                                                        end_date_time=start_date_time,
                                                        offset=offset_value,
                                                        time_delta=time_delta_value,
                                                        time_zone=time_zone)
    if isinstance(download_type, ProcurementCalculationType):
        download_type_str = str(download_type.value) if download_type != ProcurementCalculationType.ALL else None
    else:
        download_type_str = str(download_type)
    key_value_params = {RESULT_TYPE_KEY: download_type_str,
                        POINT_RESOLUTION_KEY: point_resolution,
                        QUANTILE_START_KEY: start_x,
                        QUANTILE_STEP_KEY: step_y,
                        QUANTILE_STOP_KEY: stop_z,
                        VERSION_KEY: version_number}
    key_value_params = {x: y for x, y in key_value_params.items() if y is not None and y != ''}
    data_query = dict_to_and_or_query(value_dict=key_value_params, key_name='match')
    if custom_query and isinstance(custom_query, dict):
        data_query = merge_queries(custom_query, data_query)
    data_dataframe = get_data_from_elastic_by_time(start_time_value=start_time,
                                                   end_time_value=end_time,
                                                   elastic_query=data_query,
                                                   elastic_index=elastic_index,
                                                   dict_to_flat=True,
                                                   time_interval_key=time_keys)
    return data_dataframe


def filter_calculation_results_by_type(input_data: pandas.DataFrame,
                                       result_type: ProcurementCalculationType = None,
                                       variable_columns: list = None,
                                       fixed_columns: list | str = None,
                                       key_column: str = 'version',
                                       key_value: str | int = None):
    """
    Filters input dataframe by result type and version number

    :param input_data: input dataframe
    :param result_type: ATC or NCPB
    :param variable_columns: keywords for unique instances
    :param fixed_columns: group by these columns
    :param key_column: column name to use for filter to latest
    :param key_value:  if specified, filters to this, latest by default
    :return: filtered dataframe
    """
    if result_type is not None:
        type_data = input_data[input_data['result_type'] == result_type.value]
    else:
        type_data = input_data
    if type_data.empty:
        return type_data
    type_data = type_data.dropna(axis=1, how='all')
    if key_value is None:
        type_data = filter_dataframe_to_latest_by_key(input_dataframe=type_data,
                                                      variable_columns=variable_columns,
                                                      fixed_columns=fixed_columns,
                                                      key_column=key_column)
    else:
        key_column = next(iter([column for column in type_data.columns.to_list()
                                if key_column.lower() in column.lower()]), None)
        if key_column:
            type_data = type_data[type_data[key_column] == key_value]
    filtered_data = type_data.reset_index(drop=True)
    filtered_data = delete_columns(input_dataframe=filtered_data, delete_empty=True)
    return filtered_data


def from_elastic_to_excel(start_time_value: str = PY_TIME_FROM,
                          end_time_value: str = PY_TIME_TO,
                          time_key: str = PY_OUTPUT_TIME_KEY,
                          elastic_index: str = PY_ELASTIC_INDEX,
                          elastic_query: dict = None,
                          excel_columns: list | str = PY_OUTPUT_EXCEL_COLUMNS,
                          to_minio: bool = PY_OUTPUT_FILE_TO_MINIO,
                          to_rabbit: bool = PY_OUTPUT_FILE_TO_RABBIT,
                          rabbit_exchange: str = PY_RMQ_EXCHANGE,
                          rabbit_exchange_headers: dict = PY_PROPOSED_RMQ_HEADERS,
                          receivers: list | object = None,
                          sender: object = None,
                          local_path: str = EXCEL_FOLDER_TO_STORE,
                          to_local: bool = False):
    """
    Download elastic index to Excel file and store it to location specified

    :param start_time_value: start time for the time_key if given
    :param end_time_value: end_time for time_key if given
    :param time_key: query results by time if given
    :param elastic_index: index from where to query
    :param elastic_query: custom query if needed. Note that if no time_key, time intervals are not given and no elastic
    query either then it falls back to match_all: {}
    :param excel_columns: filter dataframe/excel columns by this list if given
    :param to_minio: upload Excel file to minio
    :param to_rabbit: upload Excel file to rabbit
    :param rabbit_exchange: specify exchange where to send in rabbit
    :param rabbit_exchange_headers: specify headers
    :param receivers: receivers for rabbit (PDN) messages
    :param sender: sender for rabbit (PDN) messages (nice to have only)
    :param local_path: where to store if run locally
    :param to_local: whether to save to local
    :return: None
    """
    if PY_OUTPUT_TIME_KEY:
        received_data = get_data_from_elastic_by_time(start_time_value=start_time_value,
                                                      end_time_value=end_time_value,
                                                      elastic_query=additional_parameters,
                                                      elastic_index=elastic_index,
                                                      dict_to_flat=True,
                                                      time_interval_key=time_key)

    else:
        elastic_query = elastic_query or {"match_all": {}}
        received_data = get_data_from_elastic(elastic_index=elastic_index,
                                              elastic_query=elastic_query,
                                              dict_to_flat=True,
                                              use_default_fields=False)
    if not received_data.empty:
        to_local = to_local and is_pycharm()
        generate_excel_from_dataframes(sheets=received_data,
                                       to_minio=to_minio,
                                       to_local=to_local,
                                       to_rabbit=to_rabbit,
                                       exchange_name=rabbit_exchange,
                                       receivers=receivers,
                                       sender=sender,
                                       exchange_headers=rabbit_exchange_headers,
                                       excel_columns=excel_columns,
                                       local_path=local_path)


if __name__ == '__main__':

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    additional_parameters = None
    if isinstance(PY_OUTPUT_CUSTOM_QUERY, dict):
        additional_parameters = dict_to_and_or_query(value_dict=PY_OUTPUT_CUSTOM_QUERY, key_name='match')
    if PY_ELASTIC_TYPE is None:
        from_elastic_to_excel(start_time_value=PY_TIME_FROM,
                              end_time_value=PY_TIME_TO,
                              time_key=PY_OUTPUT_TIME_KEY,
                              elastic_index=PY_ELASTIC_INDEX,
                              elastic_query=additional_parameters,
                              excel_columns=PY_OUTPUT_EXCEL_COLUMNS,
                              to_minio=PY_OUTPUT_FILE_TO_MINIO,
                              to_rabbit=PY_OUTPUT_FILE_TO_RABBIT,
                              rabbit_exchange=PY_RMQ_EXCHANGE,
                              rabbit_exchange_headers=PY_PROPOSED_RMQ_HEADERS,
                              receivers=xml_receivers,
                              sender=xml_sender,
                              local_path=EXCEL_FOLDER_TO_STORE,
                              to_local=True)
    else:
        elastic_data = get_calculation_results_from_elastic(start_date_time=PY_TIME_FROM,
                                                            end_date_time=PY_TIME_TO,
                                                            offset_value=PY_VALID_PERIOD_OFFSET,
                                                            time_delta_value=PY_VALID_PERIOD_TIME_DELTA,
                                                            time_zone=PY_CALCULATION_TIME_ZONE,
                                                            elastic_index=PY_ELASTIC_INDEX,
                                                            download_type=PY_DOWNLOAD_TYPE,
                                                            point_resolution=PY_POINT_RESOLUTION,
                                                            start_x=PY_START_X,
                                                            step_y=PY_STEP_Y,
                                                            stop_z=PY_STOP_Z,
                                                            version_number=PY_OUTPUT_VERSION_NUMBER,
                                                            custom_query=additional_parameters)
        if (isinstance(elastic_data, pandas.DataFrame) and elastic_data.empty) or elastic_data is None:
            logger.warning(f"No matches found for the parameters requested...")
            bid_data_got = pandas.DataFrame()
            atc_data_got = pandas.DataFrame()
        else:
            bid_data_got = filter_calculation_results_by_type(input_data=elastic_data,
                                                              result_type=ProcurementCalculationType.NCPB)
            atc_data_got = filter_calculation_results_by_type(input_data=elastic_data,
                                                              result_type=ProcurementCalculationType.ATC)

        all_results = []
        if not atc_data_got.empty:
            all_results.extend(convert_to_calculation_result(input_data=atc_data_got,
                                                             download_type=ProcurementCalculationType.ATC,
                                                             sender=xml_sender,
                                                             receivers=xml_receivers))
        if not bid_data_got.empty:
            all_results.extend(convert_to_calculation_result(input_data=bid_data_got,
                                                             download_type=ProcurementCalculationType.NCPB,
                                                             sender=xml_sender,
                                                             receivers=xml_receivers))
        if all_results is not None and len(all_results) > 0:
            handle_json_output(results=all_results,
                                 prefixes=prefixes_to_use,
                                 excel_columns=PY_OUTPUT_EXCEL_COLUMNS,
                                 output_file_type=PY_OUTPUT_FILE_TYPE)
    print("Done")
