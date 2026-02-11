import copy
import itertools
import logging
import uuid
from typing import Any

import pandas

from datetime import datetime, timedelta

from py.common.functions import align_dataframe_column_names, align_country_names, \
    rename_multi_index
from py.common.df_functions import slice_data_by_time_range, get_column_names_from_data, filter_dataframe_by_keywords, \
    get_table_date_range
from py.common.time_functions import convert_string_to_datetime, convert_datetime_to_string_utc, parse_duration, \
    time_delta_to_str, get_datetime_columns_of_data_frame, set_time_zone
from py.common.ref_constants import PROCESS_TYPE_KEY, VALID_FROM_KEY, VALID_TO_KEY, DOMAIN_MRID_KEY, POINT_QUANTITY_KEY, \
    DIRECTION_CODE_KEY, OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY, TYPE_DESCRIPTION_KEY, TYPE_NAME_KEY
from py.common.to_elastic_logger import initialize_custom_logger
from py.handlers.elastic_handler import PY_PROCUREMENT_PROPOSED_INDEX, \
    PY_PROCUREMENT_ATC_INDEX, PY_PROCUREMENT_NCPB_INDEX, dict_to_and_or_query, get_data_from_elastic_by_time
from py.handlers.rabbit_handler import PY_RMQ_EXCHANGE
from py.procurement.procurement_common import get_task_from_environment, get_quantiles_from_columns, align_resolutions, \
    subtract_columns, sum_columns_by_string_in_column, AlignmentType, handle_step_data, \
    sum_columns, min_columns, max_columns, merge_tables, filter_dataframe, get_dataframe_time_resolution
from py.procurement.constants import QUANTILES_INDEX_NAME, PY_ATC_QUERY, \
    PY_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD, PY_DATA_PERIOD_TIME_DELTA, PY_OUTPUT_SEND_TO_ELASTIC, \
    PY_OUTPUT_SEND_XML_TO_MINIO, PY_OUTPUT_SEND_XLSX_TO_MINIO, PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE, \
    PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE, XML_FOLDER_TO_STORE, EXCEL_FOLDER_TO_STORE, \
    PY_NCPB_FILTER, PY_ATC_FILTER, PY_OUTPUT_SEND_XML_OUT, PY_PROPOSED_RMQ_HEADERS, PY_OUTPUT_SEND_XLSX_OUT, \
    PY_TABLE_COUNTRY_KEY, PY_EXCEL_ADD_CORRECTED_NCPB, PY_BID_QUERY, DEFAULT_AREA, \
    PY_NCPB_TIME_KEY_FROM, PY_NCPB_TIME_KEY_TO, PY_ATC_TIME_KEY_FROM, PY_ATC_TIME_KEY_TO, PY_NEGATIVE_VALUE_POLICY, \
    PY_NCPB_FILL_NA
from py.data_classes.enums import ProcurementCalculationType, InputColumnName, BusinessProductType, \
    BidDirection, NegativeValuesHandler
from py.data_classes.task_classes import ProcurementCalculationTask, QuantileArray, QuantileResult, TimeSlice, \
    TimeSliceResult, Operator
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_bid import CalculationResultBid
from py.procurement.procurement_output import handle_parsed_output

logger = logging.getLogger(__name__)

NCPB_COLUMN_KEYS = {InputColumnName.START_TIME_COLUMN: [VALID_FROM_KEY, 'start_time'],
                    InputColumnName.END_TIME_COLUMN: [VALID_TO_KEY, 'end_time'],
                    InputColumnName.REGION_COLUMN: [DOMAIN_MRID_KEY, 'in_domain.name'],
                    InputColumnName.TYPE_COLUMN: [PROCESS_TYPE_KEY, TYPE_DESCRIPTION_KEY, TYPE_NAME_KEY],
                    InputColumnName.VALUE_COLUMN: [POINT_QUANTITY_KEY, 'value.value'],
                    InputColumnName.DIRECTION_COLUMN: [DIRECTION_CODE_KEY, 'direction'],
                    InputColumnName.TYPE_DIRECTION_COLUMN: ['type.description',
                                                            'Bid_TimeSeries.standard_MarketProduct.marketProductType']}
ATC_COLUMN_KEYS = {InputColumnName.START_TIME_COLUMN: [VALID_FROM_KEY, 'utc_start'],
                   InputColumnName.END_TIME_COLUMN: [VALID_TO_KEY, 'utc_end'],
                   InputColumnName.VALUE_COLUMN: [POINT_QUANTITY_KEY, 'value.value', 'quantity', 'Point.quantity'],
                   InputColumnName.REGION_OUT_COLUMN: [OUT_DOMAIN_MRID_KEY, 'TimeSeries.out_Domain.mRID'],
                   InputColumnName.REGION_IN_COLUMN: [IN_DOMAIN_MRID_KEY, 'TimeSeries.in_Domain.mRID']}

DEFAULT_INDEX_KEYS = [InputColumnName.START_TIME_COLUMN, InputColumnName.END_TIME_COLUMN]
DEFAULT_NCPB_GROUP_KEYS = {InputColumnName.REGION_COLUMN: DEFAULT_AREA,
                           InputColumnName.TYPE_COLUMN: [BusinessProductType.mFRR, BusinessProductType.aFRR],
                           InputColumnName.DIRECTION_COLUMN: [BidDirection.Upward, BidDirection.Downward]}
DEFAULT_ATC_GROUP_KEYS = {InputColumnName.REGION_OUT_COLUMN: DEFAULT_AREA,
                          InputColumnName.REGION_IN_COLUMN: DEFAULT_AREA}
DEFAULT_VALUE_KEY = [InputColumnName.VALUE_COLUMN]

MERGE_INPUT_TABLES = 'tables'


def get_data_table(input_data,
                   index_columns: list | str,
                   type_columns: list | str,
                   value_column: list | str,
                   replacement_dict: dict = None,
                   filter_dict: dict = None,
                   agg_function: str| str = 'sum',
                   heading_keys: list = None,
                   sum_keys: list = None,
                   fill_na_value: Any = None,
                   **kwargs):
    """
    Prepares available values pivot table

    :param filter_dict: additional filtering before pivoting
    :param replacement_dict: column -> replacements, replace values in dictionary
    :param agg_function: what kind of function to be used for pivoting
    :param sum_keys: use this to sum columns
    :param heading_keys: use this to reorder key names
    :param type_columns: columns to specify unique combinations
    :param index_columns: columns to set index (time values)
    :param value_column: column where to take value
    :param input_data: dataframe with raw data
    :param fill_na_value: fill na values in dataframe
    :return: pivoted (and filtered) table
    """
    if filter_dict is not None:
        initial_data = filter_dataframe_by_keywords(input_data=input_data, filter_dict=filter_dict)
    else:
        initial_data = copy.deepcopy(input_data)
    if replacement_dict is not None:
        for k, v in replacement_dict.items():
            if k in initial_data.columns and isinstance(v, dict):
                initial_data[k] = initial_data[k].replace(v)
    output = pandas.pivot_table(initial_data,
                                values=value_column,
                                index=index_columns,
                                columns=type_columns,
                                aggfunc=agg_function)
    if sum_keys is not None:
        output = sum_columns_by_string_in_column(input_dataframe=output, string_list=sum_keys)
    if heading_keys is not None:
        output = align_dataframe_column_names(input_dataframe=output, key_words=heading_keys)
    if fill_na_value is not None:
        output = output.fillna(fill_na_value)
    return output


def calculate_double_weight_period(input_data: pandas.DataFrame,
                                   slice_column: str | tuple,
                                   end_time_moment: datetime,
                                   double_weight_period: str | timedelta = PY_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD):
    """
    Duplicates last x days of data in dataframe

    :param input_data: input data frame
    :param slice_column: datetime column by which the double weighed period is filtered
    :param end_time_moment: specify endpoint to where to double weight
    :param double_weight_period: specify double weight period in days
    :return: updated dataframe
    """
    if isinstance(double_weight_period, str):
        double_weight_period = parse_duration(double_weight_period)
    double_weight_start = end_time_moment - double_weight_period
    double_weight_end = end_time_moment
    double_weight_df = slice_data_by_time_range(data=input_data,
                                                time_ranges=(double_weight_start, double_weight_end),
                                                column_to_slice=slice_column)
    input_data = pandas.concat([input_data.reset_index(), double_weight_df.reset_index()])
    return input_data


def calculate_quantiles(input_data: pandas.DataFrame,
                        spacing: list | float,
                        interpolation_method: str = 'lower',
                        index_column_name: str = QUANTILES_INDEX_NAME):
    """
    Calculates the quantile table from input (all columns included)

    :param input_data: input dataframe
    :param spacing: list or float for quantiles
    :param interpolation_method: specify the method used for interpolation
    :param index_column_name: name of the index of output
    :return: dataframe of quantiles (index + all columns)
    """
    internal_column_name = 'positive'
    quantile_values = get_quantiles_from_columns(input_data=input_data,
                                                 quantile_spacing=spacing,
                                                 interpolation_method=interpolation_method,
                                                 axis_name=internal_column_name).reset_index()
    quantile_values[index_column_name] = round(100 * (1 - quantile_values[internal_column_name]), 2)
    quantile_values = quantile_values.sort_index(axis=1)
    quantile_values = quantile_values.drop(columns=[internal_column_name])
    quantile_values = quantile_values.sort_values(by=[index_column_name],
                                                  ascending=False).set_index(index_column_name)
    for column_name in quantile_values.columns.to_list():
        quantile_values[column_name] = (quantile_values[column_name]
                                             .where(quantile_values[column_name] >= 0, 0))
    return quantile_values


def slice_input_data(input_data: pandas.DataFrame, time_slice: TimeSlice, slice_column):
    """
    Slices dataframe by time using the slice_column

    :param input_data: input dataframe
    :param time_slice: TimeSliceInstance (uses star_time and end_time)
    :param slice_column: column by which to slice (Note that only one can be used)
    :return: sliced data
    """
    if not time_slice.start_time or not time_slice.end_time:
        return input_data
    if slice_column not in input_data.columns.to_list():
        input_data.reset_index()
    # Set timezone to one in time slice:
    try:
        input_data[slice_column] = input_data[slice_column].dt.tz_localize('UTC')
    except TypeError:
        pass
    input_data[slice_column] = input_data[slice_column].dt.tz_convert(time_slice.time_zone)
    input_data = input_data.set_index(slice_column)
    input_data = input_data.between_time(start_time=time_slice.start_time, end_time=time_slice.end_time)
    input_data = input_data.reset_index()
    return input_data


def calculate_multiple_quantiles(input_data: pandas.DataFrame,
                                 quantile_spacings: list[QuantileArray],
                                 quantile_calculation_type: ProcurementCalculationType,
                                 interpolation_method: str = 'lower',
                                 index_column_name: str = QUANTILES_INDEX_NAME):
    """
    Calculates the quantile tables from input (all columns included)

    :param quantile_calculation_type: Enum to differentiate data origins
    :param input_data: input dataframe
    :param quantile_spacings: list of different quantile instances
    :param interpolation_method: specify the method used for interpolation
    :param index_column_name: name of the index of output
    :return: list QuantileResult items
    """
    if not isinstance(quantile_spacings, list):
        return None
    output = []
    for quantile_spacing in quantile_spacings:
        quantile_result = calculate_quantiles(input_data=input_data,
                                              spacing=quantile_spacing.spacing,
                                              interpolation_method=interpolation_method,
                                              index_column_name=index_column_name)
        output.append(QuantileResult(quantile_array=quantile_spacing,
                                     calculation_type=quantile_calculation_type,
                                     quantile_result=quantile_result))
    return output


def calculate_time_slice_quantiles(input_data: pandas.DataFrame,
                                   time_slices: list[TimeSlice],
                                   time_start_column,
                                   time_end_column,
                                   quantile_calculation_type: ProcurementCalculationType,
                                   quantiles_spacing: list[QuantileArray]):
    """
    Calculates multiple quantile tables based on the time slices and quantiles variations

    :param input_data: input dataframe
    :param time_slices: data about slicing the data
    :param time_start_column: column where time period start is defined
    :param time_end_column: column where time period end is defined
    :param quantile_calculation_type: ProcurementCalculationType for determining the results afterward.
    :param quantiles_spacing: quantile arrays
    :return:
    """
    time_slice_results = []
    for time_slice in time_slices:
        sliced_df = slice_input_data(input_data = input_data,
                                     time_slice=time_slice,
                                     slice_column=time_start_column)
        sliced_df = sliced_df.reset_index(drop=True)
        sliced_df = sliced_df.sort_index(axis=1)
        sliced_df = sliced_df.drop(columns=[time_start_column, time_end_column])
        sliced_df = sliced_df.sample(frac=1).reset_index(drop=True)
        slice_quantities = calculate_multiple_quantiles(input_data=sliced_df,
                                                        quantile_calculation_type=quantile_calculation_type,
                                                        quantile_spacings=quantiles_spacing)
        for slice_quantity in slice_quantities:
            time_slice_results.append(TimeSliceResult.from_quantile_result(quantile_result=slice_quantity,
                                                                           time_slice_result=time_slice))
    return time_slice_results


def handle_single_query(data_period_start_time: datetime | str,
                        data_period_end_time: datetime | str,
                        input_index: str,
                        single_query: dict,
                        index_columns: list | str,
                        type_columns: list,
                        value_columns: list | str,
                        time_start_key: str = PY_NCPB_TIME_KEY_FROM,
                        time_end_key: str = PY_NCPB_TIME_KEY_TO,
                        mapping: dict = None,
                        group_by_filter: dict = None,
                        fill_value: Any = None,
                        **kwargs):
    """
    Function for handling single query

    :param data_period_start_time: start time from when to start to ask data
    :param data_period_end_time: end time to when to ask data
    :param input_index: Elastic index from where to query data
    :param type_columns: columns to specify unique combinations
    :param index_columns: columns to set index (time values)
    :param value_columns: column where to take value
    :param single_query: query for Elastic
    :param time_start_key: column name which contains start time for the valid period
    :param time_end_key: column name which contains end time for the valid period
    :param mapping: mapping for getting keys for pivot
    :param group_by_filter: additional filters if rows need to be grouped
    :param fill_value: whether to fill step data
    :return: pivoted dataframe and dataframe with initial data
    """
    index_columns = [index_columns] if isinstance(index_columns, str) else index_columns
    type_columns = [type_columns] if isinstance(type_columns, str) else type_columns
    value_columns = [value_columns] if isinstance(value_columns, str) else value_columns
    mapping = mapping or NCPB_COLUMN_KEYS
    query_for_data = dict_to_and_or_query(value_dict=single_query, key_name='match')
    data_got = get_data_from_elastic_by_time(start_time_value=data_period_start_time,
                                             end_time_value=data_period_end_time,
                                             elastic_index=input_index,
                                             elastic_query=query_for_data,
                                             time_interval_key =time_start_key)
    data_got = set_time_zone(input_data=data_got, columns=[time_start_key, time_end_key],
                             time_zone='UTC')
    data_mapping = get_column_names_from_data(input_data=data_got, input_mapping=mapping)
    if group_by_filter is not None:
        data_got = filter_dataframe(input_data=data_got, mapping_keys=data_mapping, filter_set=group_by_filter)
    index_col = [data_mapping.get(x) for x in index_columns]
    type_col = [data_mapping.get(x) for x in type_columns]
    value_col = [data_mapping.get(x) for x in value_columns]
    value_col = value_col[0] if len(value_col) == 1 else value_col
    if fill_value is not None:
        data_got = handle_step_data(input_data=data_got,
                                    time_columns=index_col,
                                    query_start_time=data_period_start_time,
                                    query_end_time=data_period_end_time,
                                    type_columns=type_col)
    data_pivot = pandas.DataFrame()
    if value_col is not None or (isinstance(value_col, list) and len(value_col) > 1):
        data_pivot = get_data_table(input_data=data_got,
                                    index_columns=index_col,
                                    type_columns=type_col,
                                    value_column=value_col,
                                    **kwargs)
    return data_pivot, data_got


def get_matching_combination(input_list: list, combinations: list, delimiters: list = None):
    """
    For translating dataframe columns to common format. Filters input list by possible combinations

    :param delimiters: use this to split into substrings and check each of them individually
    :param input_list: list to check
    :param combinations: possible matches
    :return: input_list elements as keys and found combinations as values
    """
    delimiters = delimiters or ['_', ' ']
    output = {}
    for input_val in input_list:
        for tuple_val in combinations:
            found = []
            for y in input_val:
                for x in tuple_val:
                    delimiter = next(iter([x for x in delimiters if x in y]), None)
                    sets = y.split(delimiter) if delimiter is not None else [y]
                    z = None
                    for s in sets:
                        if hasattr(x, 'value_of') and callable(x.value_of):
                            try:
                                z = x.value_of(s)
                                break
                            except ValueError:
                                pass
                    if z is not None and z not in found:
                        found.append(z)
            if len(found) == len(tuple_val):
                output[input_val] = found
    return output


def is_sublist(sub_list, main_list):
    """
    Checks if list of strings are present in another list in ordered matter. Loose version of list being a sublist of
    another list

    :param sub_list: list to check
    :param main_list: where sublist should be
    :return: true if all elements were found in order specified, False otherwise
    """
    s_l = len(sub_list)
    m_l  = len(main_list)
    if s_l > m_l:
        return False
    for i in range(m_l - s_l + 1):
        is_in = True
        for j in range(s_l):
            is_in = is_in and sub_list[j] in main_list[i + j]
        if is_in:
            return  True
    return False


def merge_atc(tables: list,  atc_country_area_codes: list = None, **kwargs):
    """
    Special custom function for merging ATC data. Takes 1D ATC, subtracts Intraday flows for same border and adds the
    last to opposite direction. NB! works currently only with 2 tables

    :param tables: list of pivoted dataframes
    :param atc_country_area_codes:  list of areas for which borders are needed
    :param kwargs: for bypassing additional parameters
    :return:
    """
    atc_pivot = pandas.DataFrame
    if len(tables) == 2:
        day_ahead_df, id_schedules = tables
        merged_atc = day_ahead_df.ffill().merge(id_schedules.ffill(),
                                                left_index=True,
                                                right_index=True,
                                                how='outer',
                                                suffixes=('_1D', '_ID'))
        existing_pairs = day_ahead_df.columns.to_list()
        all_border_pairs = [(x, y) for x in atc_country_area_codes for y in atc_country_area_codes if x != y]
        same_border_pairs = [[x for x in existing_pairs if all([z in x for z in y])] for y in all_border_pairs]
        same_border_pairs = [list(y) for y in set(tuple(x) for x in [z for z in same_border_pairs if z != []])]
        borders = []
        c_list = merged_atc.columns.to_list()
        for border_pair in same_border_pairs:
            for i in range(len(border_pair)):
                ahead = border_pair[i]
                other = border_pair[i + 1 if i < len(border_pair) - 1 else 0]
                d_plus = next(iter([x for x in c_list if is_sublist(ahead, x) and '_1D' in x[0]]))
                i_minus = next(iter([x for x in c_list if is_sublist(ahead, x) and '_ID' in x[0]]))
                op_plus = next(iter([x for x in c_list if is_sublist(other, x) and '_ID' in x[0]]))
                merged_atc[ahead] = merged_atc[d_plus] - merged_atc[i_minus] + merged_atc[op_plus]
                borders.append(ahead)
        atc_pivot = merged_atc[borders]
    return atc_pivot


QUERY_OPERATORS = {'ADD': Operator(method=merge_tables, parameters={'merge_method': sum_columns}),
                   'SUBTRACT': Operator(method=merge_tables, parameters={'merge_method': subtract_columns}),
                   'ATC_SUBTRACT': Operator(method=merge_atc),
                   'MIN': Operator(method=merge_tables, parameters={'merge_method': min_columns}),
                   'MAX': Operator(method=merge_tables, parameters={'merge_method': max_columns})}



def get_operation(input_dict: dict, operators: dict = None):
    """
    Gets operator from operators dictionary based on the keyword in input dict

    :param operators: dictionary with keywords and corresponding operators
    :param input_dict: Query dictionary
    :return: found operator or None
    """
    operators = operators or QUERY_OPERATORS
    if len(input_dict.keys()) == 1:
        first_key = next(iter(input_dict.keys()))
        if not isinstance(input_dict[first_key], list):
            return None
        return operators.get(first_key)
    return None


def deal_with_negative_values(input_data: pandas.DataFrame(), method: NegativeValuesHandler = PY_NEGATIVE_VALUE_POLICY):
    """
    Deal with negative values within input data

    :param input_data: input dataframe (pivoted)
    :param method: policy for handling the negative values
    :return: updated dataframe
    """
    if method == NegativeValuesHandler.CUT_NEGATIVE:
        input_data = input_data[(input_data >= 0).all(axis=1)]
    elif method == NegativeValuesHandler.NEGATIVE_TO_ZERO:
        input_data[input_data < 0] = 0
    return input_data


def change_table_naming(input_data: pandas.DataFrame, combinations: list):
    """
    Switches dataframe names to common format

    :param input_data: pivoted dataframe
    :param combinations: name combinations
    :return: updated dataframe
    """
    matches = get_matching_combination(input_list=input_data.columns.tolist(), combinations=combinations)
    new_m = {k: tuple([x.name if x.name is not None else x.value for x in v]) for k, v in matches.items()}
    input_data.columns = rename_multi_index(input_data.columns, new_m)
    input_data = input_data[new_m.values()]
    return input_data


def handle_queries(input_query: dict,
                   data_period_start_time: datetime | str,
                   data_period_end_time: datetime | str,
                   input_index: str,
                   mapping: dict,
                   index_columns: list,
                   type_columns: dict,
                   value_columns: list,
                   time_start_key: str = PY_NCPB_TIME_KEY_FROM,
                   time_end_key: str = PY_NCPB_TIME_KEY_TO,
                   group_by_filter: dict = None,
                   fill_value: str = None,
                   **kwargs):
    """
    Handles queries by operation


    :param input_query: query or dictionary with operator and set of queries
    :param data_period_start_time: start time from when to start to ask data
    :param data_period_end_time: end time to when to ask data
    :param input_index: Elastic index from where to query data
    :param time_start_key: column name which contains start time for the valid period
    :param time_end_key: column name which contains end time for the valid period
    :param mapping: mapping to use and filter the column names
    :param index_columns: index columns for pivoting
    :param type_columns: columns for unique types
    :param value_columns: columns for values
    :param group_by_filter: dictionary with additional filters if needed
    :param fill_value: fill na values in data
    :return: collected data
    """
    combinations = list(itertools.product(*type_columns.values()))
    operator = get_operation(input_dict=input_query)
    resolution = None
    if operator is not None:
        queries = next(iter(input_query.values()))
        initial_data = []
        pivoted = []
        for single_query in queries:
            single_pivot, single_data, resolution = handle_queries(input_query=single_query,
                                                                   data_period_start_time=data_period_start_time,
                                                                   data_period_end_time=data_period_end_time,
                                                                   input_index=input_index,
                                                                   time_start_key=time_start_key,
                                                                   time_end_key=time_end_key,
                                                                   mapping=mapping,
                                                                   index_columns=index_columns,
                                                                   type_columns=type_columns,
                                                                   value_columns=value_columns,
                                                                   group_by_filter=group_by_filter,
                                                                   fill_value=fill_value,
                                                                   # fill_na_value=0,
                                                                   **kwargs)
            pivoted.append(single_pivot)
            initial_data.append(single_data)
        if len(pivoted) > 1:
            # Escape empty dataframes
            pivoted = [x for x in pivoted if isinstance(x, pandas.DataFrame) and not x.empty]
            tables, resolution = align_resolutions(inputs=pivoted, alignment_type=AlignmentType.TO_LOWER)
            merge_function = operator.method or merge_tables
            merge_f_args = {**kwargs, **{MERGE_INPUT_TABLES: tables}}
            merge_f_args = {**merge_f_args, **operator.parameters} if operator.parameters is not None else merge_f_args
            pivot_data = merge_function(**merge_f_args)
        else:
            pivot_data = pivoted[0]
        initial_data = pandas.concat(initial_data)
    else:
        pivot_data, initial_data = handle_single_query(data_period_start_time=data_period_start_time,
                                                       data_period_end_time=data_period_end_time,
                                                       input_index=input_index,
                                                       single_query=input_query,
                                                       time_start_key=time_start_key,
                                                       time_end_key=time_end_key,
                                                       mapping= mapping,
                                                       index_columns=index_columns,
                                                       type_columns=list(type_columns.keys()),
                                                       value_columns=value_columns,
                                                       group_by_filter =group_by_filter,
                                                       fill_value=fill_value,
                                                       **kwargs)
        p_t_columns = get_datetime_columns_of_data_frame(input_data=pivot_data.reset_index())
        if not pivot_data.empty:
            resolution = get_dataframe_time_resolution(input_data=pivot_data, time_columns=p_t_columns)
            pivot_data = change_table_naming(input_data=pivot_data, combinations=combinations)

    return pivot_data, initial_data, resolution


def estimate_non_procured_bids(calculation_task: ProcurementCalculationTask = None,
                               bid_query: dict = PY_BID_QUERY,
                               data_period_start_time: str | datetime = None,
                               data_period_end_time: str | datetime = None,
                               bid_input_index: str = PY_PROCUREMENT_NCPB_INDEX,
                               data_period: str | timedelta = PY_DATA_PERIOD_TIME_DELTA,
                               double_weight_period: str | timedelta = PY_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD,
                               group_by_filter: list | dict | str = PY_NCPB_FILTER,
                               time_key_from: str = PY_NCPB_TIME_KEY_FROM,
                               time_key_to: str = PY_NCPB_TIME_KEY_TO,
                               table_country_key: str = PY_TABLE_COUNTRY_KEY,
                               negative_values_method: NegativeValuesHandler = PY_NEGATIVE_VALUE_POLICY,
                               fill_value: str = PY_NCPB_FILL_NA,
                               query_offset: str = 'P1D',
                               quantiles_spacing: list[QuantileArray] = None,
                               time_slices: list[TimeSlice] = None,
                               bid_mapping: dict =None,
                               bid_index_columns: list =None,
                               bid_type_columns: dict = None,
                               bid_value_columns: list = None):
    """
    Calculates available bids probabilities

    :param time_key_from: column name for from data
    :param time_key_to: column name for to data
    :param table_country_key: country key for Excel tables
    :param bid_query: query for specify bids
    :param group_by_filter: additional dictionary with column names and functions (basically min max
    :param time_slices: time slices if specified
    :param calculation_task: instance containing data about the calculation
    :param data_period_start_time: start of calculation
    :param data_period_end_time: end of calculation
    :param bid_input_index: bids data index
    :param data_period: period of data
    :param double_weight_period: period for double weight
    :param quantiles_spacing: spacing for calculation
    :param bid_mapping: mapping to use and filter the column names
    :param bid_index_columns: index columns for pivoting
    :param bid_type_columns: columns for unique types
    :param bid_value_columns: columns for values
    :param negative_values_method: how to handle negative values
    :param fill_value: fill dataframe if needed
    :param query_offset: for querying data which is presented as steps give large enough offset to get left endpoint
    :return: quantiles
    """
    bid_mapping = bid_mapping or NCPB_COLUMN_KEYS
    bid_index_columns= bid_index_columns or DEFAULT_INDEX_KEYS
    bid_type_columns = bid_type_columns or DEFAULT_NCPB_GROUP_KEYS
    bid_value_columns = bid_value_columns or DEFAULT_VALUE_KEY
    process_type_change = {PROCESS_TYPE_KEY: {BusinessProductType.SA_mFRR.value: BusinessProductType.mFRR.value,
                                              BusinessProductType.DA_mFRR.value: BusinessProductType.mFRR.value,
                                              BusinessProductType.CA_aFRR.value: BusinessProductType.aFRR.value,
                                              BusinessProductType.LA_aFRR.value: BusinessProductType.aFRR.value}}

    if calculation_task:
        data_period_start_time = data_period_start_time or calculation_task.data_period_start
        data_period_end_time = data_period_end_time or calculation_task.data_period_end
        quantiles_spacing = quantiles_spacing or calculation_task.spacings
        time_slices = time_slices or calculation_task.time_slices
    # 1. Query data
    default_offset = parse_duration(query_offset)
    corrected_start_time = convert_string_to_datetime(data_period_start_time) - default_offset
    bids_pivot, initial_data, resolution = handle_queries(input_query=bid_query,
                                                           data_period_start_time=corrected_start_time,
                                                           data_period_end_time=data_period_end_time,
                                                           input_index=bid_input_index,
                                                           time_start_key=time_key_from,
                                                           time_end_key=time_key_to,
                                                           mapping=bid_mapping,
                                                           replacement_dict=process_type_change,
                                                           index_columns=bid_index_columns,
                                                           type_columns=bid_type_columns,
                                                           value_columns=bid_value_columns,
                                                           fill_value=fill_value,
                                                           group_by_filter=group_by_filter)
    bids_pivot = deal_with_negative_values(input_data=bids_pivot, method=negative_values_method)
    resolution_string = time_delta_to_str(resolution)
    bids_pivot = align_country_names(input_dataframe=bids_pivot,
                                      area_list=calculation_task.lfc_areas,
                                      attribute_name=table_country_key)
    time_keys = get_datetime_columns_of_data_frame(input_data=bids_pivot.reset_index())
    time_start_column = time_keys[0] if len(time_keys) > 0 else None
    time_end_column = time_keys[1] if len(time_keys) > 1 else None
    bids_pivot = slice_data_by_time_range(data=bids_pivot,
                                         time_ranges=(data_period_start_time, data_period_end_time),
                                         column_to_slice=time_start_column)
    # 6. Get and add double-weighed period
    data_period_start_time, data_period_end_time = get_table_date_range(input_data=bids_pivot,
                                                                        from_column=time_start_column,
                                                                        to_column=time_end_column,
                                                                        start_time_moment=data_period_start_time,
                                                                        end_time_moment=data_period_end_time,
                                                                        data_time_period=data_period)
    if calculation_task:
        calculation_task.data_period_start = data_period_start_time
        calculation_task.data_period_end = data_period_end_time
    final_pivot_original = copy.deepcopy(bids_pivot)
    bids_pivot = calculate_double_weight_period(input_data=bids_pivot,
                                                slice_column=time_start_column,
                                                end_time_moment=data_period_end_time,
                                                double_weight_period=double_weight_period)
    # 7. Calculate quantiles
    time_slice_results = calculate_time_slice_quantiles(input_data=bids_pivot,
                                                        time_slices=time_slices,
                                                        quantiles_spacing=quantiles_spacing,
                                                        quantile_calculation_type=ProcurementCalculationType.NCPB,
                                                        time_start_column=time_start_column,
                                                        time_end_column=time_end_column)
    return CalculationResultBid(calculation_type=ProcurementCalculationType.NCPB,
                                data_period_start=data_period_start_time,
                                data_period_end=data_period_end_time,
                                task=calculation_task,
                                pivoted_data=final_pivot_original.reset_index(),
                                input_data=initial_data,
                                time_resolution=resolution_string,
                                quantiles=time_slice_results)


def estimate_atc(calculation_task: ProcurementCalculationTask = None,
                 data_period_start_time: str | datetime = None,
                 data_period_end_time: str | datetime = None,
                 atc_index: str = PY_PROCUREMENT_ATC_INDEX,
                 atc_query: dict | list = PY_ATC_QUERY,
                 time_key_from: str = PY_ATC_TIME_KEY_FROM,
                 time_key_to: str = PY_ATC_TIME_KEY_TO,
                 data_period: str | timedelta = PY_DATA_PERIOD_TIME_DELTA,
                 double_weight_period: str | timedelta = PY_DATA_PERIOD_DOUBLE_WEIGHT_PERIOD,
                 group_by_filter: list | dict |str = PY_ATC_FILTER,
                 table_country_key: str = PY_TABLE_COUNTRY_KEY,
                 time_slices: list[TimeSlice] = None,
                 negative_values_method: NegativeValuesHandler = PY_NEGATIVE_VALUE_POLICY,
                 query_offset: str = 'P30D',
                 quantiles_spacing: list[QuantileArray] = None,
                 atc_country_area_codes: list = None,
                 atc_mapping: dict = None,
                 atc_index_columns: list = None,
                 atc_type_columns: dict = None,
                 atc_value_columns: list = None
                 ):
    """
    Calculates available atc probabilities

    :param table_country_key: country key for Excel tables
    :param group_by_filter: additional dictionary with column names and functions (basically min max
    :param time_slices: time slices if specified
    :param calculation_task: instance containing data about the calculation
    :param data_period_start_time: start of calculation
    :param data_period_end_time: end of calculation
    :param atc_index: atc data index
    :param atc_query: query for specify atc
    :param time_key_from: column name for from data
    :param time_key_to: column name for to data
    :param data_period: period of data
    :param double_weight_period: period for double weight
    :param quantiles_spacing: spacing vector for atc calculation
    :param atc_mapping: mapping to use and filter the column names
    :param atc_index_columns: index columns for pivoting
    :param atc_type_columns: columns for unique types
    :param atc_value_columns: columns for values
    :param atc_country_area_codes: list of area codes
    :param negative_values_method: how to handle negative values
    :param query_offset: for querying data which is presented as steps give large enough offset to get left endpoint
    :return: quantiles
    """
    atc_mapping = atc_mapping or ATC_COLUMN_KEYS
    atc_index_columns= atc_index_columns or DEFAULT_INDEX_KEYS
    atc_type_columns = atc_type_columns or DEFAULT_ATC_GROUP_KEYS
    atc_value_columns = atc_value_columns or DEFAULT_VALUE_KEY

    if calculation_task:
        data_period_start_time = data_period_start_time or calculation_task.data_period_start
        data_period_end_time = data_period_end_time or calculation_task.data_period_end
        quantiles_spacing = quantiles_spacing or calculation_task.spacings
        time_slices = time_slices or calculation_task.time_slices
    # 1. Query data
    default_offset = parse_duration(query_offset)
    corrected_start_time = convert_string_to_datetime(data_period_start_time) - default_offset
    if not atc_country_area_codes:
        atc_country_area_codes = [area.name for area in calculation_task.lfc_areas]
    atc_pivot, initial_data, resolution = handle_queries(input_query=atc_query,
                                                         data_period_start_time=corrected_start_time,
                                                         data_period_end_time=data_period_end_time,
                                                         input_index=atc_index,
                                                         time_start_key=time_key_from,
                                                         time_end_key=time_key_to,
                                                         mapping=atc_mapping,
                                                         index_columns=atc_index_columns,
                                                         type_columns=atc_type_columns,
                                                         value_columns=atc_value_columns,
                                                         atc_country_area_codes=atc_country_area_codes,
                                                         group_by_filter=group_by_filter)
    atc_pivot = deal_with_negative_values(input_data=atc_pivot, method=negative_values_method)
    time_keys = get_datetime_columns_of_data_frame(input_data=atc_pivot.reset_index())
    time_start_column = time_keys[0] if len(time_keys) > 0 else None
    time_end_column = time_keys[1] if len(time_keys) > 1 else None
    atc_pivot = slice_data_by_time_range(data=atc_pivot,
                                         time_ranges=(data_period_start_time, data_period_end_time),
                                         column_to_slice=time_start_column)
    resolution_string = time_delta_to_str(resolution)
    atc_pivot = align_country_names(input_dataframe=atc_pivot,
                                      area_list=calculation_task.lfc_areas,
                                      attribute_name=table_country_key)
    # 6. Get and add double-weighed period
    data_period_start_time, data_period_end_time = get_table_date_range(input_data=atc_pivot,
                                                                        from_column=time_start_column,
                                                                        to_column=time_end_column,
                                                                        start_time_moment=data_period_start_time,
                                                                        end_time_moment=data_period_end_time,
                                                                        data_time_period=data_period)
    if calculation_task:
        calculation_task.data_period_start = data_period_start_time
        calculation_task.data_period_end = data_period_end_time
    atc_pivot_original = copy.deepcopy(atc_pivot)
    atc_pivot = calculate_double_weight_period(input_data=atc_pivot,
                                               slice_column=time_start_column,
                                               end_time_moment=data_period_end_time,
                                               double_weight_period=double_weight_period)
    time_slice_results = calculate_time_slice_quantiles(input_data=atc_pivot,
                                                        time_slices=time_slices,
                                                        quantiles_spacing=quantiles_spacing,
                                                        quantile_calculation_type=ProcurementCalculationType.ATC,
                                                        time_start_column=time_start_column,
                                                        time_end_column=time_end_column)

    return CalculationResultATC(calculation_type=ProcurementCalculationType.ATC,
                                data_period_start=data_period_start_time,
                                data_period_end=data_period_end_time,
                                task=calculation_task,
                                pivoted_data=atc_pivot_original.reset_index(),
                                time_resolution=resolution_string,
                                input_data=initial_data,
                                quantiles=time_slice_results)


if __name__ == '__main__':

    initialize_custom_logger(extra_fields={"Job": "Procurement Calculation", "Job_id": str(uuid.uuid4())})
    task = get_task_from_environment()

    logger.info(f"Data is taken from {convert_datetime_to_string_utc(task.data_period_start)} "
                f"until {convert_datetime_to_string_utc(task.data_period_end)}")
    logger.info(f"Calculation is valid from {convert_datetime_to_string_utc(task.valid_from)} "
                f"until {convert_datetime_to_string_utc(task.valid_to)}")

    all_results = []
    calculation_type = task.calculation_type
    if calculation_type == ProcurementCalculationType.NCPB or calculation_type == ProcurementCalculationType.ALL:
        logger.info(f"Forecast for {str(ProcurementCalculationType.NCPB.value)}")
        all_results.append(estimate_non_procured_bids(calculation_task=task))
    if calculation_type == ProcurementCalculationType.ATC or calculation_type == ProcurementCalculationType.ALL:
        logger.info(f"Forecast for {str(ProcurementCalculationType.ATC.value)}")
        all_results.append(estimate_atc(calculation_task=task))




    handle_parsed_output(results=all_results,
                         output_to_elastic=PY_OUTPUT_SEND_TO_ELASTIC,
                         elastic_index=PY_PROCUREMENT_PROPOSED_INDEX,
                         output_xml_minio=PY_OUTPUT_SEND_XML_TO_MINIO,
                         output_xml_out=PY_OUTPUT_SEND_XML_OUT,
                         output_xml_local=PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE,
                         rabbit_exchange=PY_RMQ_EXCHANGE,
                         rabbit_headers=PY_PROPOSED_RMQ_HEADERS,
                         xml_local_path=XML_FOLDER_TO_STORE,
                         output_xlsx_minio=PY_OUTPUT_SEND_XLSX_TO_MINIO,
                         output_xlsx_rabbit=PY_OUTPUT_SEND_XLSX_OUT,
                         output_xlsx_local=PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE,
                         add_corrected_ncpb=PY_EXCEL_ADD_CORRECTED_NCPB,
                         xlsx_local_path=EXCEL_FOLDER_TO_STORE)
    print("Done")
