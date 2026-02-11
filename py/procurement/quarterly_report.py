import copy
import datetime
import logging
import sys

import pandas

from py.common.functions import calculate_start_and_end_date, align_country_names
from py.data_classes.results.result_functions import handle_dataframe_timezone_excel
from py.data_classes.results.calculation_result_main import CalculationResult
from py.data_classes.task_classes import TimeSliceResult
from py.handlers.elastic_handler import PY_PROCUREMENT_REALISED_INDEX, dict_to_and_or_query, \
    PY_PROCUREMENT_PROPOSED_INDEX
from py.parsers.json_to_calculation_result import get_calculation_results_from_elastic, \
    filter_calculation_results_by_type, convert_to_calculation_result, is_pycharm
from py.parsers.parser_constants import PY_OUTPUT_CUSTOM_QUERY, PY_TIME_TO, PY_TIME_FROM, PY_OUTPUT_FILE_TO_MINIO, \
    PY_TIME_OFFSET, PY_TIME_DELTA
from py.procurement.constants import PY_TABLE_COUNTRY_KEY, EXCEL_FOLDER_TO_STORE, DEFAULT_AREA
from py.procurement.procurement_common import get_areas
from py.procurement.procurement_output import generate_excel_from_dataframes
from py.procurement.realised_data_check import ProcurementCalculationType

logger = logging.getLogger(__name__)


def get_corrected_data(start_time: str | datetime.datetime = None,
                       end_time: str | datetime.datetime = None,
                       result_type: ProcurementCalculationType = ProcurementCalculationType.ALL,
                       elastic_index: str = PY_PROCUREMENT_REALISED_INDEX,
                       custom_query: dict = None,
                       areas: list = None):
    """
    Gets NCPB data and 'corrects' it with ATC data (min over value and borders)

    :param elastic_index: index from where to query the data
    :param areas: list of LFC areas, if not provided, then default will be used (needed for NCPB correction by ATC)
    :param start_time: start time from where to get data
    :param end_time: end time to where to get data (today midnight will be default)
    :param result_type: NCPB, ATC or both
    :param custom_query: if needed limit amount fo responses with this
    :return: list of CalculationResultBid instances
    """
    areas = areas or DEFAULT_AREA
    received_data = get_calculation_results_from_elastic(start_date_time=start_time,
                                                         end_date_time=end_time,
                                                         elastic_index=elastic_index,
                                                         download_type=result_type,
                                                         custom_query=custom_query)
    bid_data_received = pandas.DataFrame()
    atc_data_received = pandas.DataFrame()
    if isinstance(received_data, pandas.DataFrame) and not received_data.empty:
        bid_data_received = filter_calculation_results_by_type(input_data=received_data,
                                                               result_type=ProcurementCalculationType.NCPB)
        atc_data_received = filter_calculation_results_by_type(input_data=received_data,
                                                               result_type=ProcurementCalculationType.ATC)
    atc_data = None
    bid_data = None
    if not atc_data_received.empty:
        atc_data = convert_to_calculation_result(input_data=atc_data_received,
                                                 download_type=ProcurementCalculationType.ATC)
    if not bid_data_received.empty:
        bid_data= convert_to_calculation_result(input_data=bid_data_received,
                                                 download_type=ProcurementCalculationType.NCPB)
    for single_bid in bid_data:
        for single_atc in atc_data:
            single_bid.update_ncpb_by_atc(atc_results=single_atc, areas=areas)
    return bid_data


def get_matching_quantile(results: list | CalculationResult, time_slice: TimeSliceResult):
    """
    Gets matching TimeSlice from the results. Note that it only checks the time space

    :param results: CalculationResult instance or list of them
    :param time_slice: TimeSlice for which to search the value
    :return:
    """
    results = [results] if isinstance(results, CalculationResult) else results
    for single_result in results:
        for single_time_slice in single_result.quantiles:
            if time_slice.same_time_slice(single_time_slice):
                return single_time_slice
    return None


def get_report_data(start_time: str | datetime.datetime = None,
                    end_time: str | datetime.datetime = None,
                    result_type: ProcurementCalculationType = ProcurementCalculationType.ALL,
                    areas: list = None,
                    custom_query: dict = None):
    """
    Generates (quarterly report) for presenting.

    :param start_time: start time from where to get data
    :param end_time: end time to where to get data (today midnight will be default)
    :param result_type: NCPB, ATC or both
    :param custom_query: if needed limit amount fo responses with this
    :param areas: list of areas
    :return: dataframe containing proposed and realised data
    """
    areas = areas or DEFAULT_AREA
    if custom_query is not None and isinstance(custom_query, dict):
        custom_query = dict_to_and_or_query(value_dict=custom_query, key_name='match')
    proposed_bids = get_corrected_data(start_time=start_time,
                                       end_time=end_time,
                                       result_type=result_type,
                                       elastic_index=PY_PROCUREMENT_PROPOSED_INDEX,
                                       custom_query=custom_query,
                                       areas=areas)
    realised_bids = get_corrected_data(start_time=start_time,
                                       end_time=end_time,
                                       result_type=result_type,
                                       elastic_index=PY_PROCUREMENT_REALISED_INDEX,
                                       custom_query=custom_query,
                                       areas=areas)
    all_results = []
    for realised_bid in realised_bids:
        for real_quantile in realised_bid.quantiles:
            real_df = copy.deepcopy(real_quantile.quantile_result)
            real_df.columns = pandas.MultiIndex.from_tuples([(*c, 'realised') for c in real_df.columns])
            i_len = next(iter([len(x) for x in real_df.columns.to_list()]))
            sender = realised_bid.sender
            if sender is not None and hasattr(sender, PY_TABLE_COUNTRY_KEY):
                sender = getattr(sender, PY_TABLE_COUNTRY_KEY)
            if not real_df.empty:
                real_df.loc[:, ('valid_from', *[''] * (i_len - 1))] = real_quantile.time_slice.valid_from
                real_df.loc[:, ('valid_to', *[''] * (i_len - 1))] = real_quantile.time_slice.valid_to
                real_df.loc[:, ('TSO', *[''] * (i_len - 1))] = sender
                prop_quantile = get_matching_quantile(results=proposed_bids, time_slice=real_quantile)
                if prop_quantile is not None:
                    prop_df = copy.deepcopy(prop_quantile.quantile_result)
                    prop_df.columns = pandas.MultiIndex.from_tuples([(*c, 'proposed') for c in prop_df.columns])
                    real_df = real_df.merge(prop_df, how='left',  left_index=True, right_index=True).reset_index()
                all_results.append(real_df)
    final_dataframe = pandas.concat(all_results).reset_index(drop=True)
    df_cols = final_dataframe.columns.tolist()
    ordered_list = ['valid_from', 'valid_to', 'TSO', 'percentage_level']
    type_ids = {ordered_list.index(y): df_cols.index(x) for x in df_cols for z in x for y in ordered_list if y in z}
    type_cols = [df_cols[type_ids[x]] for x in sorted(type_ids.keys())]

    final_dataframe = final_dataframe.set_index(type_cols)
    if isinstance(areas, list):
        final_dataframe = align_country_names(input_dataframe=final_dataframe,
                                              area_list=areas,
                                              attribute_name=PY_TABLE_COUNTRY_KEY)
    final_dataframe = final_dataframe.reindex(sorted(final_dataframe.columns), axis=1)
    final_dataframe = final_dataframe.reset_index()
    final_dataframe = final_dataframe.sort_values(by=type_cols).reset_index(drop=True)
    return final_dataframe


if __name__ == '__main__':

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    from_time, to_time = calculate_start_and_end_date(start_date_time=PY_TIME_FROM,
                                                      end_date_time=PY_TIME_TO,
                                                      offset=PY_TIME_OFFSET,
                                                      time_delta=PY_TIME_DELTA,
                                                      default_timedelta='P4M')
    report_areas = get_areas()
    report = get_report_data(start_time=from_time,
                             end_time= to_time,
                             # data_offset=PY_TIME_OFFSET,
                             # time_delta=PY_TIME_DELTA,
                             areas=report_areas,
                             result_type=ProcurementCalculationType.ALL,
                             custom_query= PY_OUTPUT_CUSTOM_QUERY)

    report = handle_dataframe_timezone_excel(input_data=report)
    if not report.empty:
        to_local = True and is_pycharm()
        time_format = "%Y%m%dT%H%M"
        components = ['SPT_report',
                      'from', from_time.strftime(time_format),
                      'to',  to_time.strftime(time_format),
                      'at', datetime.datetime.now().strftime(time_format)]
        file_name = f"{'_'.join(components)}.xlsx"
        generate_excel_from_dataframes(sheets=report,
                                       to_minio=PY_OUTPUT_FILE_TO_MINIO,
                                       to_local=to_local,
                                       full_file_name=file_name,
                                       # to_rabbit=PY_OUTPUT_FILE_TO_RABBIT,
                                       # exchange_name=PY_RMQ_EXCHANGE,
                                       # exchange_headers=PY_PROPOSED_RMQ_HEADERS,
                                       # receivers=xml_receivers,
                                       # sender=xml_sender,
                                       local_path=EXCEL_FOLDER_TO_STORE,
                                       )
