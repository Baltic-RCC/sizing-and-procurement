import copy
import logging
import os
import time
from collections.abc import Iterable
from datetime import datetime
from io import BytesIO
from pathlib import Path
import EDX
import pandas
from brcc_apis.rabbit import RMQConsumer

from py.common.functions import save_bytes_io_to_local, get_random_string_length
from py.data_classes.enums import ProcurementCalculationType, OutputFileType
from py.data_classes.results.calculation_result_main import CalculationResult
from py.data_classes.task_classes import merge_object_lists_by_keys
from py.handlers.elastic_handler import PY_ELASTICSEARCH_HOST, PY_PROCUREMENT_PROPOSED_INDEX
from py.handlers.rabbit_handler import PY_RMQ_EXCHANGE, send_data_to_rabbit, BlockingClient, ProcurementBlockingClient, \
    EDXHeader
from py.procurement.constants import (PY_XML_PREFIX, PY_OUTPUT_SEND_XML_TO_MINIO, \
                                      PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE, XML_FOLDER_TO_STORE,
                                      PY_OUTPUT_SEND_XLSX_TO_MINIO, \
                                      PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE, EXCEL_FOLDER_TO_STORE,
                                      PY_PROPOSED_RMQ_HEADERS,
                                      PY_OUTPUT_SEND_XML_OUT, METADATA_KEY, PY_EXCEL_ADD_INPUT_DATA, \
                                      PY_EXCEL_ADD_OBJECT_LIST, PY_EXCEL_ADD_QUANTILE_DATA, PY_EXCEL_ADD_PIVOTED_DATA,
                                      CORRECTED_NCPB_KEY,PY_PROCUREMENT_STARTING_PERCENT,PY_PROCUREMENT_ENDING_PERCENT,PY_OUTPUT_SEND_WITH_RABBIT,PY_OUTPUT_SEND_WITH_EDX,PY_PROCUREMENT_SEND_SIZE)
from py.handlers.minio_handler import save_file_to_minio_with_link

import config
from py.common.config_parser import parse_app_properties
parse_app_properties(globals(), config.paths.config.pdn)

import zipfile

logger = logging.getLogger(__name__)

# EDX, when sending to itself, runs to internal infinite loop if more than 2 messages are sent
RMQ_SLEEP_TIME = 15
if PY_RMQ_EXCHANGE == "procurement":
    RMQ_SLEEP_TIME = 5


def results_to_xmls(result: CalculationResult,
                    to_minio: bool = PY_OUTPUT_SEND_XML_TO_MINIO,
                    to_local: bool = PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE,
                    to_out: bool = PY_OUTPUT_SEND_XML_OUT,
                    local_path: str = XML_FOLDER_TO_STORE,
                    exchange_name: str = PY_RMQ_EXCHANGE,
                    exchange_headers: dict = PY_PROPOSED_RMQ_HEADERS,
                    prefixes: list = None):
    """
    Auxiliary function for getting xml documents

    :param rabbit_client: rabbit instance
    :param exchange_headers: headers for the rabbit client
    :param exchange_name: rabbit exchange name
    :param to_rabbit: whether to send output to rabbit
    :param prefixes: prefixes to add in front of file names
    :param result: calculation results
    :param to_minio: boolean to save to minio
    :param to_local: boolean to save to local storage
    :param local_path: path in local storage
    :return: None
    """
    if prefixes is None:
        prefixes = PY_XML_PREFIX
    if to_minio or to_local or to_out:

        result_xmls = []
        result_xmls.extend(result.generate_xml(additional_prefixes=prefixes))

        for result_xml in result_xmls:
            if to_minio:
                logger.info(f"Saving {result_xml.name} to Minio")
                save_file_to_minio_with_link(result_xml)
            if to_local:
                logger.info(f"Saving {result_xml.name} to local storage")
                save_bytes_io_to_local(bytes_io_object=result_xml, location_name=local_path)
            if to_out:
                logger.info(f"Sending  {result_xml.name} to {exchange_name}")

                send_to_out(payload=result_xml,
                            exchange_name=exchange_name,
                            headers=exchange_headers,
                            file_extension=OutputFileType.XML)
                if RMQ_SLEEP_TIME is not None:
                    time.sleep(int(RMQ_SLEEP_TIME))


def get_addresses_from_results(result: CalculationResult):
    """
    Gets list of receivers and sender from list of CalculationResult instances

    :param result: list of calculation results
    :return: tuple consisting of list of receiver addresses and sender address
    """
    receivers = []
    senders = []
    receivers = merge_object_lists_by_keys(old_list=receivers, new_list=result.receivers, primary_key='mRID')
    sender = [result.sender] if not isinstance(result.sender, list) else result.sender
    senders = merge_object_lists_by_keys(old_list=senders, new_list=sender, primary_key='mRID')
    return receivers, senders[0]


def save_dataframe_to_excel(input_df, excel_writer, sheet_name: str, save_header_separately: bool =False):
    """
    Saves multilevel dataframe to excel. Header and body separately. If single level then goes with default function

    :param save_header_separately: True then removes header from body and save both (skips empty line
    :param input_df: input dataframe
    :param excel_writer: excel writer instance
    :param sheet_name: sheet name
    :return:
    """
    if save_header_separately and isinstance(input_df.columns, pandas.MultiIndex):
        header = input_df.columns.to_frame(index=False)
        header = header.T.reset_index()
        data_df = input_df.set_axis(range(len(input_df.columns)), axis=1).reset_index()
        header.to_excel(excel_writer, sheet_name=sheet_name, merge_cells=True, header=False, index=False)
        data_df.to_excel(excel_writer, sheet_name=sheet_name, merge_cells=True, header=False, index=False,
                         startrow=len(header.index))
    else:
        input_df.to_excel(excel_writer, sheet_name=sheet_name, merge_cells=True)


def generate_excel_from_dataframes(sheets: dict | list | pandas.DataFrame,
                                   full_file_name: str = None,
                                   to_minio: bool = False,
                                   to_local: bool = False,
                                   to_out: bool = False,
                                   exchange_name: str = None,
                                   receivers: list | object = None,
                                   sender: object = None,
                                   exchange_headers: dict = None,
                                   excel_columns: str | list = None,
                                   local_path: str = EXCEL_FOLDER_TO_STORE):
    """
    Auxiliary function for getting xml documents

    :param sheets: list or dict of pandas Dataframes or single dataframe
    :param full_file_name: file name for Excel
    :param exchange_headers: headers for rabbit
    :param exchange_name: name of the rabbit exchange
    :param to_rabbit: send to rabbit
    :param receivers: list or objects that contain receiver attribute
    :param sender: object that contains receiver attribute
    :param to_minio: boolean to save to minio
    :param to_local: boolean to save to local storage
    :param excel_columns: specify excel columns to keep
    :param rabbit_client: rabbit instance
    :param local_path: path in local storage
    :return: None
    """
    excel_columns = [excel_columns] if isinstance(excel_columns, str) else excel_columns
    excel_file = BytesIO()
    sheets = [sheets] if isinstance(sheets, pandas.DataFrame) else sheets
    sheets = {f"Sheet{x + 1}": dataframe for x, dataframe in enumerate(sheets)} if isinstance(sheets, list) else sheets
    full_file_name = (full_file_name or
                      f"generated_at_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_by_spt_service.xlsx")

    with pandas.ExcelWriter(excel_file) as writer:
        for sheet_name, data_table in sheets.items():
            if excel_columns is not None:
                cols_to_keep = [x for x in data_table.columns.to_list()
                                if any(x for y in excel_columns if str(y).lower() == str(x).lower())]
                if cols_to_keep is not None and len(cols_to_keep) > 0:
                    data_table = data_table[cols_to_keep]
                else:
                    logger.warning(f"Dataframe has any of the columns given, skipping filtering by columns")

            try:
                sheet_name = sheet_name.replace(':', '-')
                save_dataframe_to_excel(input_df=data_table, excel_writer=writer, sheet_name=sheet_name)
                """zip_file_name = full_file_name[:-5]
                        print(zip_file_name)
                        zip_file = zipfile.ZipFile(zip_file_name, 'w')
                        zip_file.write(full_file_name)
                        zip_file.close()"""
            except Exception as error:
                print(error)
    excel_file.name = full_file_name
    if to_minio:
        logger.info(f"Saving {full_file_name} to Minio")
        save_file_to_minio_with_link(excel_file)
    if to_local:
        logger.info(f"Saving {full_file_name} to local storage")
        save_bytes_io_to_local(bytes_io_object=excel_file, location_name=local_path)
    if to_out and receivers is not None and exchange_name is not None:
        receivers = [receivers] if not isinstance(receivers, Iterable) else receivers
        for receiver in receivers:
            # Workaround for dummy tsos
            message_id = '-'.join([Path(full_file_name).stem, receiver.mRID]) if hasattr(receiver, 'mRID') else None
            logger.info(f"Sending  {full_file_name} to {exchange_name}")
            send_to_out(payload=excel_file,
                               message_id=message_id,
                               exchange_name=exchange_name,
                               headers=exchange_headers,
                               sender=sender,
                               receiver=receiver,
                               file_extension=OutputFileType.XLSX)
            if RMQ_SLEEP_TIME is not None:
                time.sleep(int(RMQ_SLEEP_TIME))


def results_to_xlsx(result: CalculationResult,
                    to_minio: bool = PY_OUTPUT_SEND_XLSX_TO_MINIO,
                    to_local: bool = PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE,
                    to_out: bool = PY_OUTPUT_SEND_XML_OUT,
                    add_input_data: bool = PY_EXCEL_ADD_INPUT_DATA,
                    add_object_list: bool = PY_EXCEL_ADD_OBJECT_LIST,
                    add_quantile_data: bool = PY_EXCEL_ADD_QUANTILE_DATA,
                    add_pivoted_data: bool = PY_EXCEL_ADD_PIVOTED_DATA,
                    exchange_name: str = PY_RMQ_EXCHANGE,
                    exchange_headers: dict = PY_PROPOSED_RMQ_HEADERS,
                    prefixes: list = None,
                    excel_columns: str | list = None,
                    local_path: str = EXCEL_FOLDER_TO_STORE):
    """
    Auxiliary function for getting xml documents

    :param add_pivoted_data: add pivoted data to excel
    :param add_quantile_data: add quantiles data to excel
    :param add_object_list: convert and add object list to excel
    :param add_input_data: add input data to excel
    :param exchange_headers: headers for rabbit
    :param exchange_name: name of the rabbit exchange
    :param to_rabbit: send to rabbit
    :param result: calculation results
    :param to_minio: boolean to save to minio
    :param to_local: boolean to save to local storage
    :param excel_columns: specify excel columns to keep
    :param local_path: path in local storage
    :param prefixes: prefixes to add in front of file names
    :param rabbit_client: rabbit instance
    :return: None
    """
    random_nr = 5
    if prefixes is None:
        prefixes = PY_XML_PREFIX
    if to_minio or to_local or to_out:
        sheets = {}
        if hasattr(result, 'NCPB_updated') and getattr(result, 'NCPB_updated'):
            new_sheets = result.generate_xlsx_sheets(add_quantile_data=add_quantile_data)
        else:
            new_sheets = result.generate_xlsx_sheets(add_input_data=add_input_data,
                                                        add_object_list=add_object_list,
                                                        add_quantile_data=add_quantile_data,
                                                        add_pivoted_data=add_pivoted_data)
        sheets = {**sheets, **new_sheets}
        #result_types = list(x.calculation_type.value for x in results)
        full_file_name = copy.deepcopy(prefixes)
        #full_file_name.extend(result_types)
        full_file_name.append(datetime.now().strftime("%d-%m-%Y"))
        full_file_name.append(get_random_string_length(random_nr))
        full_file_name = '-'.join(list(dict.fromkeys(full_file_name))) + '.xlsx'
        receivers, sender = get_addresses_from_results(result=result)
        generate_excel_from_dataframes(sheets=sheets,
                                       full_file_name=full_file_name,
                                       to_minio=to_minio,
                                       to_local= to_local,
                                       to_out=to_out,
                                       exchange_name=exchange_name,
                                       receivers=receivers,
                                       sender=sender,
                                       exchange_headers=exchange_headers,
                                       excel_columns=excel_columns,
                                       local_path=local_path)


def results_to_elastic(result: CalculationResult,
                       elastic_index: str = PY_PROCUREMENT_PROPOSED_INDEX,
                       elastic_server: str = PY_ELASTICSEARCH_HOST):
    """
    Auxiliary function for sending to elastic

    :param result: list of CalculationResult instances
    :param elastic_index: elastic index where to send
    :param elastic_server: elastic server where to send
    :return: None
    """
    result.send_results_to_elastic(elastic_server=elastic_server, elastic_index=elastic_index)


def send_to_out(payload,
                       sender: object | str = None,
                       receiver: object | str = None,
                       file_extension: OutputFileType | str = None,
                       message_id: str = None,
                       message_correlation_id: str = None,
                       business_type: str = None,
                       exchange_name: str = PY_RMQ_EXCHANGE,
                       headers: dict = PY_PROPOSED_RMQ_HEADERS):
    """
    Sends content to rabbit exchange (edx). Note that edx allows in total 6 parameters from which 2 are required:
    receiverCode and businessType. 4 are additional: senderApplication, baMessageID, baCorrelationID, fileExtension.
    Here if fields are not given then suitable values are searched from headers.
    if no baMessageId and file_extension are given but payload has name (file) then ba_message_id is file name and
    file_extension is file extension

    :param business_type: specify businessType field. Business type, required
    :param message_correlation_id: specify baCorrelationID. Sender BA's message correlation identification, not required
    :param message_id: specify baMessageID field. Sender BA's message identification, not required
    :param file_extension:specify fileExtension field. Extension of file, not required
    :param receiver: specify receiverCode field. Receiver's EDX identification, required.
    :param sender: specify senderApplication field. Name of sender BA. not required
    :param payload: content to be sent
    :param exchange_name: name of the rabbit exchange
    :param headers: additional headers for the content
    :param rabbit_client: rabbit instance
    :return: None
    """
    headers = copy.deepcopy(headers) if headers is not None else headers
    if hasattr(payload, METADATA_KEY):
        try:
            headers = {**headers, **getattr(payload, METADATA_KEY)}
        except AttributeError:
            pass
    file_name = getattr(payload, 'name') if hasattr(payload, 'name') else None
    if file_name is not None:
        headers['file_name'] = payload.name
        message_id = message_id or Path(file_name).stem
        file_extension = file_extension or Path(file_name).suffix.upper()
    headers_instance = EDXHeader(fileExtension=file_extension,
                                 senderApplication=sender,
                                 receiverCode=receiver,
                                 baMessageID=message_id,
                                 businessType=business_type,
                                 baCorrelationID=message_correlation_id,
                                 additional_content=headers)
    if "edx" in str(PY_RMQ_EXCHANGE).lower():
        out_headers = headers_instance.headers_to_edx()
    else:
        out_headers = headers_instance.to_dict()
    logger.debug(f"Message headers: {out_headers}")
    if PY_OUTPUT_SEND_WITH_RABBIT:
        rabbit_client = ProcurementBlockingClient()
        send_data_to_rabbit(input_data=payload,
                            rabbit_exchange=exchange_name,
                            rabbit_client=rabbit_client,
                            headers=out_headers)
        rabbit_client.close()
    if PY_OUTPUT_SEND_WITH_EDX:
        service_edx = EDX.create_client(server=PDN_SERVER, username=PDN_USERNAME,
                                        password=PDN_PASSWORD)
        message_id = service_edx.send_message(
                        receiver_EIC=out_headers['receiverCode'],
                        business_type=out_headers['businessType'],
                        content=payload.getvalue()
                    )
        logger.debug(f"Message sent via EDX: {message_id}")




def is_pycharm():
    return os.getenv("PYCHARM_HOSTED") is not None


def handle_parsed_output(results: list,
                         prefixes: list = None,
                         excel_columns: list = None,
                         output_to_elastic: bool = False,
                         elastic_index: str = None,
                         output_xml_minio: bool = False,
                         output_xml_out: bool = False,
                         output_xml_local: bool = False,
                         rabbit_exchange: str = None,
                         rabbit_headers: dict = None,
                         xml_local_path: str = XML_FOLDER_TO_STORE,
                         output_xlsx_minio: bool = False,
                         output_xlsx_rabbit: bool = False,
                         output_xlsx_local: bool = True,
                         add_corrected_ncpb: bool = False,
                         xlsx_local_path: str = EXCEL_FOLDER_TO_STORE):
    """
    General entrypoint for handling outputs (send results to elastic, xml and/or xlsx to minio/rabbit/local)

    :param results: list of CalculationResult instances
    :param prefixes: prefixes to be added to file names
    :param excel_columns: specify excel columns if output excel needs filtering
    :param output_to_elastic: if True then converts CalculationResult.object_list to json and sends to elastic
    :param elastic_index: specify elastic index where send results
    :param output_xml_minio: if True then initiates CalculationResult.generate_xml and sends output to minio
    :param output_xml_rabbit:  if True then initiates CalculationResult.generate_xml and sends output to rabbit
    :param output_xml_local:  if True then initiates CalculationResult.generate_xml and saves output to local storage
    :param rabbit_exchange: specify rabbit exchange
    :param rabbit_headers: add additional rabbit headers
    :param xml_local_path: path where to store xml files
    :param output_xlsx_minio: if true then initiates CalculationResult.generate_xlsx_sheets and sends output to minio
    :param output_xlsx_rabbit: if true then initiates CalculationResult.generate_xlsx_sheets and sends output to rabbit
    :param output_xlsx_local: if true then initiates CalculationResult.generate_xlsx_sheets and saves to local storage
    :param add_corrected_ncpb: if true then corrects CalculationResultBid with CalculationResultATC
    :param xlsx_local_path: path where to store xlsx files
    :return: None
    """

    #sending to elastic all objects calculated in 0.1
    for result in results:
        if output_to_elastic:
            results_to_elastic(result=result, elastic_index=elastic_index)

    #FEATURE: sending to xml only filtered dataset based on inclusion list

    in_range = list(range(int(PY_PROCUREMENT_STARTING_PERCENT[0]),int(PY_PROCUREMENT_ENDING_PERCENT[0])+int(PY_PROCUREMENT_SEND_SIZE[0]),int(PY_PROCUREMENT_SEND_SIZE[0])))

    filtered_results = []

    for result in results:
        # 1. Filter the data into a temporary list structure first
        cleaned_object_list = []
        #XML results filtering
        for sublist in result.object_list:
            # Keep only objects that match the range
            valid_items = [
                obj for obj in sublist
                if hasattr(obj, 'percentage_level') and obj.percentage_level.value in in_range
            ]
            # Only add the sublist if it's not empty
            if valid_items:
                cleaned_object_list.append(valid_items)

        #Excel results filtering
        for sublist in result.quantiles:
            sublist.quantile_result=sublist.quantile_result[sublist.quantile_result.index.isin(in_range)]

        # 2. Only if we have matching data, create a new result object
        if cleaned_object_list:
            # We use deepcopy on the INNER items to ensure independence
            # but we create the TOP level object fresh to avoid the "no setter" error
            new_result = copy.deepcopy(result)
            try:
                if hasattr(new_result,'_object_list'):
                    setattr(new_result,'_object_list',cleaned_object_list)
                else:

                    new_result.__dict__['object_list'] = cleaned_object_list
            except Exception:
                pass

            filtered_results.append(new_result)
    # FEATURE-END

    for result_filtered in filtered_results:
        #if result.object_list[0][0].percentage_level.value
        result_filtered.quantiles[0].quantile_array.spacing_step_size=5 # redefine for sending, use param here
        results_to_xmls(result=result_filtered,
                        prefixes=prefixes,
                        to_minio=output_xml_minio,
                        to_local=output_xml_local and is_pycharm(),
                        to_out=output_xml_out,
                        exchange_name=rabbit_exchange,
                        exchange_headers=rabbit_headers,
                        local_path=xml_local_path)

    #based on the agreement seperatly send in excel the available bid info with ATC included
    if add_corrected_ncpb and len(filtered_results) == 2:
        atc_result = next(iter(x for x in filtered_results if x.calculation_type == ProcurementCalculationType.ATC), None)
        ncpb_result = next(iter(x for x in filtered_results if x.calculation_type == ProcurementCalculationType.NCPB), None)
        if ncpb_result is not None and atc_result is not None:
            corrected_ncpb_result = copy.deepcopy(ncpb_result)
            corrected_ncpb_result.excel_sheet_prefix = CORRECTED_NCPB_KEY
            corrected_ncpb_result.update_ncpb_by_atc(atc_result)
            filtered_results.clear()
            filtered_results = corrected_ncpb_result
            results_to_xlsx(result=filtered_results,
                            prefixes=prefixes,
                            excel_columns=excel_columns,
                            to_minio=output_xlsx_minio,
                            to_local=output_xlsx_local and is_pycharm(),
                            to_out=output_xml_out,
                            exchange_name=rabbit_exchange,
                            exchange_headers=rabbit_headers,
                            local_path=xlsx_local_path)
