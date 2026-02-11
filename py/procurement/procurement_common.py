import logging
import re
from enum import Enum, auto
from typing import Any

import numpy
import pandas
from datetime import datetime, time

from py.handlers.elastic_handler import ElkHandler, PY_ELASTICSEARCH_HOST, PY_AREA_INDEX
from py.common.functions import is_nested_dict
from py.common.df_functions import resample_by_time
from py.common.time_functions import convert_string_to_time, convert_string_to_datetime, parse_timezone, parse_duration, \
    str_to_datetime, get_time_period_from_dataframe_column, time_delta_to_str, get_datetime_columns_of_data_frame
from py.common.ref_constants import INTERPOLATE, MEAN_KEYWORD, INTERPOLATE_PARAMETERS, SERIES_VALID_FROM_F_KEY, \
    SERIES_VALID_TO_F_KEY, POINT_RESOLUTION_KEY, VALID_FROM_KEY, VALID_TO_KEY
from py.data_classes.task_classes import ProcurementCalculationTask, TypeCodes, QuantileArray, EICArea, TimeSlice
from py.procurement.constants import (CALCULATED_COUNTRIES, DEFAULT_QUERY, FINAL_SENDER, DEFAULT_AREA, DEFAULT_BLOCK, \
                                      PY_PROCUREMENT_STARTING_PERCENT, PY_PROCUREMENT_STEP_SIZE,
                                      PY_PROCUREMENT_ENDING_PERCENT, PY_DATA_PERIOD_TIME_DELTA,
                                      PY_DATA_PERIOD_START_DATE, PY_DATA_PERIOD_END_DATE,
                                      PY_DATA_PERIOD_OFFSET, \
                                      PY_PERIOD_VALID_TIME_SHIFT, PY_VALID_PERIOD_TIME_DELTA,
                                      PY_VALID_PERIOD_START_DATE, PY_VALID_PERIOD_END_DATE, \
                                      PY_VALID_PERIOD_OFFSET, PY_CALCULATION_TYPE, POWER_MEASUREMENT_UNIT,
                                      PERCENTAGE_MEASUREMENT_UNIT,
                                      CURVE_TYPE, ENERGY_PRODUCT, \
                                      CODING_SCHEME, STATUS_TYPE, PY_PROCUREMENT_SENDER_RCC,
                                      PY_PROCUREMENT_RECEIVER_TSO, PY_PROCUREMENT_LFC_AREA, \
                                      PY_PROCUREMENT_LFC_BLOCK, FINAL_RECEIVER, PY_OUTPUT_NCPB_CODES,
                                      PY_OUTPUT_ATC_CODES, PY_CALCULATION_TIME_ZONE, PY_CALCULATION_START_TIME,
                                      PY_CALCULATION_END_TIME, PY_CALCULATION_STEPS, PY_TASK_DESCRIPTION)

logger = logging.getLogger(__name__)


def calculate_time_intervals(valid_to: str | datetime,
                             valid_from: str | datetime,
                             number_of_steps: int = 1,
                             time_start: str | time = None,
                             time_end: str | time = None,
                             time_zone: str = PY_CALCULATION_TIME_ZONE):
    """
    Calculates time intervals for validity interval with the number of steps

    :param time_zone: Time zone in the time slices should be (Note if data is in UTC and slice is 00:00-02:00CET)
    :param time_start: start of the slice if specified
    :param time_end: end of the slice if specified
    :param valid_to: period to which the results are valid, NB afterward max(valid_from, start_time)
    :param valid_from: period to which the results are valid, NB, afterward min(valid_to, end_time)
    :param number_of_steps: number of steps the results need to be divided
    :return: time_slices, updated valid_from, updated valid_to
    """
    output = []
    default_delta_str = 'PT15M'
    infinitely_small_unit = parse_duration('PT1S')
    tz = parse_timezone(time_zone)
    default_time_delta = parse_duration(default_delta_str)
    valid_from = convert_string_to_datetime(valid_from)
    valid_to = convert_string_to_datetime(valid_to)
    time_start = tz.localize(datetime.combine(valid_from, convert_string_to_time(time_start))) \
        if time_start else valid_from
    time_end = tz.localize(datetime.combine(valid_from, convert_string_to_time(time_end))) \
        if time_end else valid_to
    number_of_steps = number_of_steps or 1
    if time_start == time_end and number_of_steps == 1:
        logger.info(f"No slicing applied")
        return [TimeSlice(valid_from=valid_from, valid_to=valid_to)]
    time_start, time_end = (time_start, time_end) if time_end > time_start else (time_end, time_start)
    original_time_delta = time_end - time_start
    time_delta = original_time_delta / number_of_steps
    if time_delta < default_time_delta:
        time_delta_string = time_delta_to_str(time_delta)
        logger.warning(f"{time_delta_string} is smaller than {default_delta_str}. Continuing with {default_delta_str}")
        time_delta = default_time_delta
    step_start = time_start
    for i in range(number_of_steps):
        step_end = step_start + time_delta
        output.append(TimeSlice(valid_from=step_start,
                                valid_to=step_end,
                                start_time=step_start.time(),
                                end_time=(step_end - infinitely_small_unit).time(),
                                valid_period_resolution=original_time_delta,
                                point_resolution=time_delta,
                                time_zone=time_zone,
                                number_of_points=number_of_steps,
                                point=i + 1))
        step_start = step_end
    return output, time_start, time_end


def get_areas(from_local: bool = False):
    """
    Gets data about areas

    :param from_local: for debugging
    :return:
    """
    areas = None
    if not from_local:
        areas = [EICArea.init_from_dict(x)
                 for x in get_elastic_areas(countries=CALCULATED_COUNTRIES).to_dict('records')]
    if not areas:
        areas = [area for area in DEFAULT_AREA for area_name in PY_PROCUREMENT_LFC_AREA if area.value_of(area_name)]
    return areas

def get_task_from_environment(ncpb_codes: dict = PY_OUTPUT_NCPB_CODES,
                              atc_codes: dict = PY_OUTPUT_ATC_CODES):
    """
    Packages environment variables to separate object

    :param ncpb_codes: Specify codes for Bids
    :param atc_codes: Specify codes for ATC
    :return: task instance
    """
    areas = get_areas(from_local=True)
    sender = next(iter(sender for sender in FINAL_SENDER if sender.value_of(PY_PROCUREMENT_SENDER_RCC)))
    receivers = [receiver for receiver in FINAL_RECEIVER
                 for receiver_name in PY_PROCUREMENT_RECEIVER_TSO if receiver.value_of(receiver_name)]
    lfc_block = next(iter(block for block in DEFAULT_BLOCK if block.value_of(PY_PROCUREMENT_LFC_BLOCK)))

    bid_codes = TypeCodes.type_code_from_dict(ncpb_codes)
    atc_codes = TypeCodes.type_code_from_dict(atc_codes)

    task = ProcurementCalculationTask(data_period_start=PY_DATA_PERIOD_START_DATE,
                                      data_period_end=PY_DATA_PERIOD_END_DATE,
                                      data_period_timedelta=PY_DATA_PERIOD_TIME_DELTA,
                                      data_period_offset=PY_DATA_PERIOD_OFFSET,
                                      valid_from=PY_VALID_PERIOD_START_DATE,
                                      valid_to=PY_VALID_PERIOD_END_DATE,
                                      valid_period_offset=PY_VALID_PERIOD_OFFSET,
                                      valid_period_timedelta=PY_VALID_PERIOD_TIME_DELTA,
                                      period_valid_time_shift=PY_PERIOD_VALID_TIME_SHIFT,
                                      calculation_type=PY_CALCULATION_TYPE,
                                      bids_code_types=bid_codes,
                                      atc_code_types=atc_codes,
                                      calculation_time_zone=PY_CALCULATION_TIME_ZONE,
                                      power_unit_type=POWER_MEASUREMENT_UNIT,
                                      percent_unit_type=PERCENTAGE_MEASUREMENT_UNIT,
                                      curve_type=CURVE_TYPE,
                                      energy_product=ENERGY_PRODUCT,
                                      coding_scheme=CODING_SCHEME,
                                      status_type=STATUS_TYPE,
                                      description=PY_TASK_DESCRIPTION,
                                      sender=sender,
                                      receivers=receivers,
                                      lfc_areas=areas,
                                      lfc_block=lfc_block)
    task.time_slices, task.valid_from, task.valid_to = calculate_time_intervals(time_start=PY_CALCULATION_START_TIME,
                                                                                time_end=PY_CALCULATION_END_TIME,
                                                                                number_of_steps=PY_CALCULATION_STEPS,
                                                                                valid_from=task.valid_from,
                                                                                valid_to=task.valid_to)
    task.spacings = [QuantileArray(spacing_start_value=x, spacing_end_value=y, spacing_step_size=z)
                     for x in PY_PROCUREMENT_STARTING_PERCENT
                     for y in PY_PROCUREMENT_ENDING_PERCENT
                     for z in PY_PROCUREMENT_STEP_SIZE]
    return task


def get_elastic_areas(countries: list = None,
                      elastic_address: str = PY_ELASTICSEARCH_HOST,
                      area_index: str = PY_AREA_INDEX,
                      area_query: dict = None,
                      area_key: str = 'area.name'):
    """
    Gets lfc_block_name data

    :param countries: countries to search
    :param area_index: countries index
    :param area_query: query for getting countries
    :param elastic_address: elastic host
    :param area_key: column name to search
    :return: dataframe
    """
    # countries = countries or CALCULATED_COUNTRIES
    area_query = area_query or DEFAULT_QUERY
    elk_instance = ElkHandler(server=elastic_address)
    country_map = elk_instance.get_data(query=area_query,
                                        index=str(area_index).removesuffix('*') + '*',
                                        # use_default_fields=True
                                        )
    if countries is None:
        return country_map
    atc_countries = country_map.merge(pandas.DataFrame(data=countries, columns=[area_key]),
                                      on=area_key)
    return atc_countries


def get_time_time_delta_for_point(start_time: str | datetime,
                                  end_time: str | datetime,
                                  number_of_points: int):
    """
    Takes start, end time, finds difference, divides it number of points and returns the output as duration string

    :param start_time: start time of interval
    :param end_time: end time of interval
    :param number_of_points: number of points
    :return: string of duration
    """
    start_time = convert_string_to_datetime(start_time)
    end_time = convert_string_to_datetime(end_time)
    time_delta = (end_time - start_time) / number_of_points
    resolution_string = time_delta_to_str(time_delta)
    return resolution_string


def apply_group_by_filter_transform(input_data: pandas.DataFrame,
                                    group_by_filter: dict,
                                    group_by_columns: list):
    """
    Applies transform to grouped dataframe. Each key and value pair are treated separately. Logical adding is done
    afterward.

    :param input_data: input dataframe
    :param group_by_filter: dictionary where keys are the columns where to apply and values are functions to apply
    :param group_by_columns: columns to group by
    :return: grouped dataframe or initial if something went wrong
    """
    columns = input_data.columns.to_list()
    try:
        idx = None
        for key, value in group_by_filter.items():
            if key in columns:
                if value == 'idxmax':
                    input_data = input_data.loc[input_data.groupby(group_by_columns)[key].idxmax()]
                elif value == 'idxmin':
                    input_data = input_data.loc[input_data.groupby(group_by_columns)[key].idxmin()]
                else:
                    new_idx = input_data.groupby(group_by_columns)[key].transform(value) == input_data[key]
                    idx = new_idx if idx is None else idx & new_idx
        if idx is not None:
            input_data = input_data[idx]
    except Exception:
        logger.warning(f"Unable to filter dataframe")
    return input_data


def apply_group_by_filter(input_data: pandas.DataFrame,
                          group_by_filter: list | dict | str = None,
                          group_by_columns: list = None):
    """
    Applies additional filter for grouping the data (datetime, type, direction) before pivoting

    :param input_data: dataframe of input data (bids or atc)
    :param group_by_filter: filter to apply (dict: uses transform, list or string: uses query)
    :param group_by_columns: columns to group by
    :return: grouped dataframe
    """
    if group_by_filter is None or group_by_columns is None or input_data.empty:
        return input_data
    if isinstance(group_by_filter, dict):
        return apply_group_by_filter_transform(input_data=input_data,
                                               group_by_filter=group_by_filter,
                                               group_by_columns=group_by_columns)
    elif isinstance(group_by_filter, list):
        for single_query in group_by_columns:
            input_data = apply_group_by_filter(input_data=input_data,
                                               group_by_filter=single_query,
                                               group_by_columns =group_by_columns)
    return input_data.groupby(group_by_columns).apply(group_by_filter)


def get_quantiles_from_columns(input_data: pandas.DataFrame,
                               columns: list = None,
                               quantile_spacing: list | float = None,
                               interpolation_method: str = 'lower',
                               axis_name: str = 'positive'):
    """
    Main function to calculate the quantiles

    :param input_data: input dataframe
    :param columns: columns from which quantiles are calculated (one column at a time)
    :param quantile_spacing: array or single value for which quantiles are calculated (method can depend on this)
    :param interpolation_method: if needed then how to fill the caps
    :param axis_name: name of index of output
    :return: dataframe consisting of quantiles
    """
    quantiles = pandas.DataFrame()
    if isinstance(columns, list):
        columns = [columns]
    if not columns:
        columns = input_data.columns.to_list()
    for column_name in columns:
        single_column = input_data[[column_name]]
        single_column = single_column[single_column[column_name].notna()].reset_index(drop=True)
        single_column[column_name] = single_column[column_name].sort_values(ignore_index=True)
        single_quantiles = (single_column.quantile(q=quantile_spacing,
                                                   # method='table',
                                                   interpolation=interpolation_method)
                            .rename_axis(axis_name))

        if quantiles.empty:
            quantiles = single_quantiles
        else:
            quantiles = quantiles.merge(single_quantiles, left_index=True, right_index=True)
    return quantiles


def calculate_spacing(spacing_start_value: float = 90,
                      spacing_end_value: float = 100,
                      spacing_step_size: float = 0.1,
                      rounding: int = 4):
    """
    Calculates spacing vector for quantiles

    :param spacing_start_value: start value for quantile vector in %
    :param spacing_end_value:  end value for quantile vector in %
    :param spacing_step_size: step size for quantile vector in %
    :param rounding: number of decimals for the value to be rounded
    :return: array
    """
    calculated_step_size = round(spacing_step_size / 100, rounding)
    start_value = round(1 - spacing_start_value / 100, rounding)
    end_value = round(1 - spacing_end_value / 100, rounding)
    spacing = [end_value]
    number_of_steps = round((spacing_end_value - spacing_start_value) / calculated_step_size) + 1
    default_end_value = 0
    for step in range(number_of_steps):
        new_step = round(default_end_value + step * calculated_step_size, rounding)
        if end_value < new_step <= start_value:
            spacing.append(new_step)
    return spacing


class AlignmentType(Enum):
    """
    For deciding in which direction to align resolution
    """
    TO_UPPER = auto()
    TO_LOWER = auto()


def get_dataframe_time_resolution(input_data: pandas.DataFrame, time_columns: list = None):
    """
    Gets time resolution from dataframe

    :param input_data: input dataframe
    :param time_columns: if specified then calculates from them, otherwise looks them in dataframe
    :return: timedelta value
    """
    x = input_data.reset_index()
    if not time_columns:
        time_columns = get_datetime_columns_of_data_frame(input_data=x)
    return get_time_period_from_dataframe_column(input_dataframe=str_to_datetime(data=x, columns=time_columns),
                                                 data_column=time_columns)


def generate_time_array(start_date: str | datetime,
                        end_date: str | datetime,
                        timedelta: str | Any,
                        time_start_column,
                        time_end_column = None):
    """
    Generates dataframe with time values specified start and end time and resolution

    :param start_date: start datetime for the array
    :param end_date: end datetime for the array
    :param timedelta: resolution
    :param time_start_column: column name for time values
    :param time_end_column: column name for end time values
    :return: dataframes with time values
    """
    start_date = pandas.Timestamp(convert_string_to_datetime(start_date))
    end_date = pandas.Timestamp(convert_string_to_datetime(end_date))
    if start_date.tz != end_date.tz:
        end_date = end_date.tz_convert(start_date.tz)
    timedelta = parse_duration(timedelta)
    if time_end_column is not None:
        end_date = end_date - timedelta
    array = {time_start_column: pandas.date_range(start=start_date, end=end_date, freq=timedelta)}
    if time_end_column is not None:
        array[time_end_column] = array[time_start_column] + timedelta
    time_df = pandas.DataFrame(array)
    return time_df


def align_resolutions(inputs: list[pandas.DataFrame],
                      time_columns: list = None,
                      alignment_type: AlignmentType = AlignmentType.TO_LOWER):
    """
    Aligns two dataframes to same resolution (smaller to higher: mean, higher to smaller: interpolate)

    :param inputs: first dataframe
    :param time_columns: if dataframe consists of to and from columns, and they need to be re-calculate
    :param alignment_type: LOWER: Bigger resolution interpolate to lower, UPPER: mean lower resolution to bigger
    :return: updated dataframe
    """
    if not time_columns:
        time_columns = []
        for x in inputs:
            time_columns.extend(get_datetime_columns_of_data_frame(input_data=x.reset_index()))
        time_columns = list(dict.fromkeys(time_columns))

    inputs = [str_to_datetime(data=x.reset_index(), columns=time_columns) for x in inputs]
    resolutions = [get_time_period_from_dataframe_column(input_dataframe=x.reset_index(),
                                                         data_column=time_columns) for x in inputs]
    resolution, method = (min(resolutions), INTERPOLATE) \
        if alignment_type == AlignmentType.TO_LOWER else (max(resolutions), MEAN_KEYWORD)
    method_parameters = INTERPOLATE_PARAMETERS if method == INTERPOLATE else None
    outputs = [resample_by_time(data=x,
                                sampling_time=resolution,
                                index_columns=time_columns,
                                method=method,
                                method_parameters=method_parameters) for x in inputs]
    outputs = [x.set_index(time_columns) for x in outputs]
    return outputs, resolution


def sum_columns(input_data: pandas.DataFrame, output_column_name, first_column_name, second_column_name):
    """
    Sums two columns to output column name

    :param input_data: input data
    :param output_column_name: column name where to store the results
    :param first_column_name: name of the first column
    :param second_column_name: name of the second column
    :return: updated dataframe
    """
    input_data.loc[:, output_column_name] = numpy.where(pandas.notna(input_data[first_column_name]) &
                                                        pandas.notna(input_data[second_column_name]),
                                         input_data[first_column_name] + input_data[second_column_name],
                                         input_data[first_column_name].fillna(input_data[second_column_name]))
    return input_data


def subtract_columns(input_data: pandas.DataFrame, output_column_name, first_column_name, second_column_name):
    """
    Subtracts second column from first column and saves result to output column name

    :param input_data: input data
    :param output_column_name: column name where to store the results
    :param first_column_name: name of the first column
    :param second_column_name: name of the second column
    :return: updated dataframe
    """
    input_data.loc[:, output_column_name] = input_data[first_column_name] - input_data[second_column_name]
    return input_data


def min_columns(input_data: pandas.DataFrame, output_column_name, first_column_name, second_column_name):
    """
    Takes min as output from two columns

    :param input_data: input data
    :param output_column_name: column name where to store the results
    :param first_column_name: name of the first column
    :param second_column_name: name of the second column
    :return: updated dataframe
    """
    input_data.loc[:, output_column_name] = input_data[[first_column_name, second_column_name]].min(axis=1)
    return input_data


def max_columns(input_data: pandas.DataFrame, output_column_name, first_column_name, second_column_name):
    """
    Takes max as output from two columns

    :param input_data: input data
    :param output_column_name: column name where to store the results
    :param first_column_name: name of the first column
    :param second_column_name: name of the second column
    :return: updated dataframe
    """
    input_data.loc[:, output_column_name] = max(input_data[first_column_name], input_data[second_column_name])
    return input_data


def sum_columns_by_list(input_data: pandas.DataFrame, summed_columns: list, method=sum_columns):
    """
    Applies given calculation method to indicated columns

    :param input_data: input data
    :param summed_columns: list of column headings to calculate
    :param method: calculation method (currently only summing or subtracting)
    :return: updated dataframe
    """
    for sum_column in summed_columns:
        if len(sum_column) > 1:
            for i in range(len(sum_column) - 1):
                input_data = method(input_data=input_data,
                                    output_column_name=sum_column[0],
                                    first_column_name=sum_column[0],
                                    second_column_name=sum_column[i + 1])
            input_data = input_data.drop(columns=sum_column[1:])
    return input_data


def sum_columns_by_string_in_column(input_dataframe: pandas.DataFrame,
                                    string_list: list,
                                    method=sum_columns):
    """
    Makes a calculation with columns whose headings are given

    :param input_dataframe: input data
    :param string_list: list of column headings
    :param method: currently use sum_columns to sum or subtract_columns to subtract
    :return: updated dataframe where columns with string list are replaced with the column with results
    """
    column_names = input_dataframe.columns.to_list()
    column_names_to_dict = {'.'.join(column_name): column_name for column_name in column_names}
    columns_to_summed = {}
    for key, value in column_names_to_dict.items():
        for single_string in string_list:
            key = key.replace(single_string, '.')
        if key not in columns_to_summed.keys():
            columns_to_summed[key] = []
        values = columns_to_summed.get(key, [])
        values.append(value)
        columns_to_summed[key] = values
    input_dataframe = sum_columns_by_list(input_data=input_dataframe,
                                          summed_columns=list(columns_to_summed.values()),
                                          method=method)
    return input_dataframe


def fill_na_df(input_df, method: str = None):
    """
    Fill na values in dataframe

    :param input_df: input
    :param method: specify this either ffill or bfill
    :return: updated dataframe
    """
    if method is not None:
        if method == 'ffill':
            input_df = input_df.ffill()
        elif method == 'bfill':
            input_df = input_df.bfill()
    return input_df


def generate_rows(data_slice: pandas.DataFrame,
                  time_columns: list | str = None,
                  period_columns: list | str = None,
                  resolution_column: str | tuple = POINT_RESOLUTION_KEY,
                  query_start_time: str | datetime = None,
                  query_end_time: str | datetime = None):
    """
    For data forward fill missing timestamps

    :param data_slice: input dataframe
    :param time_columns: columns which present time values
    :param period_columns: columns which present step periods
    :param resolution_column: column where to get resolution
    :param query_start_time: query start time (for left endpoint)
    :param query_end_time: query end time (for right endpoint)
    :return: updated slice
    :return:
    """
    time_columns = [VALID_FROM_KEY, VALID_TO_KEY] if time_columns is None else time_columns
    period_columns = [SERIES_VALID_FROM_F_KEY, SERIES_VALID_TO_F_KEY] if period_columns is None else period_columns
    time_columns = [time_columns] if not isinstance(time_columns, list) else time_columns
    period_columns = [period_columns] if not isinstance(period_columns, list) else period_columns
    data_slice = str_to_datetime(data_slice, [*time_columns, *period_columns])
    time_start_column = time_columns[0]
    time_end_column = time_columns[1] if len(time_columns) > 1 else None
    period_start_column = period_columns[0]
    period_end_column = period_columns[1] if len(period_columns) > 1 else period_start_column
    start_time = data_slice[period_start_column].min()
    end_time = data_slice[period_end_column].max()
    if query_start_time is not None:
        start_time = max(start_time, convert_string_to_datetime(query_start_time))
    if query_end_time is not None:
        end_time = min(end_time, convert_string_to_datetime(query_end_time))
    resolutions = data_slice[resolution_column].unique().tolist()
    resolution = resolutions[0] if len(resolutions) > 0 else None
    time_array = generate_time_array(start_date=start_time,
                                     end_date=end_time,
                                     timedelta=resolution,
                                     time_start_column=time_start_column,
                                     time_end_column=time_end_column)
    new_slice = time_array.merge(data_slice, on=time_columns, how='left')
    # new_slice = new_slice.ffill()
    new_slice = new_slice.infer_objects(copy=False).ffill()
    return new_slice


def handle_step_data(input_data: pandas.DataFrame,
                     period_columns: str | list = None,
                     time_columns: list | str = None,
                     type_columns: list = None,
                     resolution_column: str | tuple = POINT_RESOLUTION_KEY,
                     query_start_time: str | datetime = None,
                     query_end_time: str | datetime = None):
    """
    For generating truncated lines in data presented as steps

    :param input_data: input dataframe
    :param period_columns: columns which present step periods
    :param time_columns:  columns which present time values
    :param type_columns:  unique identifiers
    :param resolution_column: column where to get resolution
    :param query_start_time: query start time (for left endpoint)
    :param query_end_time:  query end time (for right endpoint)
    :return: updated dataframe
    """
    period_columns = [SERIES_VALID_FROM_F_KEY, SERIES_VALID_TO_F_KEY] if period_columns is None else period_columns
    period_columns = [period_columns] if not isinstance(period_columns, list) else period_columns
    period_columns = [x for x in period_columns if x in input_data.columns]
    if len(period_columns) < 2:
        return input_data
    group_by_columns = [*period_columns, *type_columns]
    groups = input_data.groupby(group_by_columns).apply(lambda x: generate_rows(x,
                                                                                time_columns=time_columns,
                                                                                period_columns=period_columns,
                                                                                resolution_column=resolution_column,
                                                                                query_start_time=query_start_time,
                                                                                query_end_time=query_end_time),
                                                        include_groups=True)
    return groups.reset_index(drop=True)


def merge_tables(tables: list | tuple, merge_method = None, fill_value: str = None, how: str = 'inner', **kwargs):
    """
    Merges available bids and procured bids together

    :param how: inner, outer, left or right
    :param merge_method: additional function to handle columns with same name
    :param tables: list of pandas data frame
    :param fill_value: use this to fill na
    :return: merged table
    """
    m_cnt = 1
    s_word = '_suffix'
    s_p = re.escape(s_word) + ".*$"
    if len(tables) == 0:
        return pandas.DataFrame()
    if len(tables) == 1:
        return tables[0]
    output_df = tables[0]
    # output = fill_na_df(input_df=tables[0], method=fill_value)
    for i in range(len(tables) - 1):
        suffix = ('', f"{s_word}{m_cnt}")
        new_table = fill_na_df(input_df=tables[i + 1], method=fill_value)
        output_df = output_df.merge(new_table, left_index=True, right_index=True, suffixes=suffix, how=how)
        m_cnt = m_cnt + 1
    all_columns = output_df.columns.to_list()
    group_columns = {}
    for col in all_columns:
        cleaned = tuple([re.sub(s_p, '', x) for x in col]) if isinstance(col, tuple) else re.sub(s_p, '', col)
        if cleaned not in group_columns:
            group_columns[cleaned] = []
        group_columns[cleaned].append(col)
    output_df =  sum_columns_by_list(input_data=output_df,
                                     summed_columns=list(group_columns.values()),
                                     method=merge_method)
    return output_df


def filter_dataframe(input_data: pandas.DataFrame, filter_set, mapping_keys):
    """
    Applies filters to dataframe. Specify key, value pairs to apply for single column or key dictionary
    to apply grouping

    :param input_data: input dataframe
    :param filter_set: filter to be applied
    :param mapping_keys: keys for grouping columns
    :return: updated dataframe
    """
    group_columns = list(mapping_keys.values())
    if isinstance(filter_set, dict) and is_nested_dict(filter_set):
        for k, v in filter_set.items():
            if k in input_data.columns:
                input_data[k] = input_data[k].replace('nan', pandas.NA)
                data_exact = input_data[input_data[k].isna()]
                data_filter = input_data[input_data[k].notna()]
                if not data_filter.empty:
                    data_filter = apply_group_by_filter(input_data=data_filter,
                                                        group_by_filter=v,
                                                        group_by_columns=k)
                    input_data = pandas.concat([data_exact, data_filter])
    else:
        input_data = apply_group_by_filter(input_data=input_data,
                                           group_by_filter=filter_set,
                                           group_by_columns=group_columns)
    return input_data
