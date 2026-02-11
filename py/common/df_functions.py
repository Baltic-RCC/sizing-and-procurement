import copy
import math
from datetime import datetime, timedelta

import pandas
import pytz

from py.common.functions import add_value_to_dict_recursively
from py.common.ref_constants import VALID_FROM_KEY, MEAN_KEYWORD
from py.common.time_functions import convert_string_to_datetime, parse_duration, get_datetime_columns_of_data_frame, \
    str_to_datetime, get_time_period_from_dataframe_column


def parse_dataframe_to_nested_dict(input_dataframe: pandas.DataFrame, column_delimiter: str = '.'):
    """
    Parses dataframe to nested dictionary by splitting the column name by delimiter

    :param input_dataframe: dataframe to be parsed
    :param column_delimiter: delimiter used to separate strings in column name
    :return: list of nested dictionaries
    """
    output = []
    for row in input_dataframe.iterrows():
        output_dict = {}
        content = row[1]
        for series_key, series_value in content.items():
            keys = str(series_key).split(column_delimiter)
            output_dict = add_value_to_dict_recursively(dict_value=series_value, output_dict=output_dict, keys=keys)
        output.append(output_dict)
    return output


class SlicingException(ValueError):
    pass


def get_matching_columns(data: pandas.DataFrame, column_to_match: str | tuple = VALID_FROM_KEY):
    """
    Checks if columns are present in dataframe

    :param data: input dataframe
    :param column_to_match: name of the column to search
    :return: list of possible matches if found
    """
    if isinstance(column_to_match, str):
        matching_columns = [single_column_name for single_column_name in data.columns.to_list()
                            if single_column_name.lower() == column_to_match.lower()]
    else:
        matching_columns = [single_column_name for single_column_name in data.columns.to_list()
                            if single_column_name == column_to_match]
    return matching_columns


def slice_data_by_time_range(data: pandas.DataFrame,
                             time_ranges: list | tuple | dict,
                             column_to_slice: str | tuple = VALID_FROM_KEY):
    """
    Get slice from the data between given dates. Note that slicing

    :param column_to_slice: Name of the column by which the data is sliced (by default is the 'from' column)
    :param data: input dataframe consisting at least columns 'from' and 'to' which can be converted to datetime
    :param time_ranges:
    :return: sliced data
    """
    matching_columns = get_matching_columns(data=data, column_to_match=column_to_slice)
    old_index_columns = []
    if len(matching_columns) < 1:
        old_index_columns = data.index.names
        data = data.reset_index()
        matching_columns = get_matching_columns(data=data, column_to_match=column_to_slice)
    if len(matching_columns) < 1:
        raise SlicingException(f"{column_to_slice} is not present in dataframe")
    if len(matching_columns) > 1:
        raise SlicingException(f"Too many matches for {column_to_slice}: {','.join(matching_columns)}")
    slice_column = matching_columns[0]
    time_ranges_actual = []
    if isinstance(time_ranges, dict):
        time_ranges = list(time_ranges.keys()) + list(time_ranges.values())
    for time_value in time_ranges:
        try:
            time_ranges_actual.append(convert_string_to_datetime(time_value))
        except ValueError:
            pass
    if len(time_ranges_actual) == 0:
        raise SlicingException(f"Unable to unpack the time range")
    min_time_value = min(time_ranges_actual)
    max_time_value = max(time_ranges_actual)
    data[slice_column] = pandas.to_datetime(data[slice_column])
    data_in_use = data.loc[(data[column_to_slice] >= min_time_value) & (data[column_to_slice] <= max_time_value)]
    if old_index_columns:
        data_in_use = data_in_use.set_index(old_index_columns)
    return data_in_use


def get_column_names_from_data(input_data: pandas.DataFrame, input_mapping: dict):
    """
    Gets map where keys are from input mapping and values are the first found column name that matches the values

    :param input_data: input dataframe
    :param input_mapping: keys as common column names and values as possible matches
    :return: dictionary with common column names as keys and found column names as values
    """
    output_mapping = {}
    col_list = input_data.columns.to_list()
    for k, v in input_mapping.items():
        v = [v] if not isinstance(v, list) else v
        v = [str(y).lower() for y in v]
        matches = [x for x in col_list if str(x).lower() in v]
        if len(matches) > 1:
            order = {key: idx for idx, key in enumerate(v)}
            matches = sorted(matches, key=lambda x: order.get(x, len(order)))
        if len(matches) >= 1:
            output_mapping[k] = matches[0]
    return output_mapping


def filter_dataframe_by_keywords(input_data: pandas.DataFrame, filter_dict: dict):
    """
    Filters input dataframe by searching keywords in the corresponding columns

    :param input_data: input dataframe
    :param filter_dict: column names and keywords within these columns
    :return: filtered dataframe
    """
    output_data = copy.deepcopy(input_data)
    if filter_dict is not None:
        for k, v in filter_dict.items():
            if k in input_data.columns:
                input_types = output_data[k].unique().tolist()
                type_values = [v] if not isinstance(v, list) else v
                matches = [x for x in input_types for y in type_values if str(y).lower() in str(x).lower()]
                if len(matches) > 0:
                    type_df = pandas.DataFrame(data=matches, columns=[k])
                    output_data = output_data.merge(type_df, on=k)
    return output_data


def filter_by_nan(input_data: pandas.DataFrame, column_name: str, filter_value):
    """
    For filtering values that are null

    :param input_data: input dataframe
    :param column_name: column to be filter
    :param filter_value: value to filter, if nan is given then returns isnan()
    :return: filtered dataframe
    """
    if not isinstance(filter_value, str) and math.isnan(filter_value):
        return input_data[input_data[column_name].isna()]
    return input_data[input_data[column_name] == filter_value]


def filter_dataframe(input_data: pandas.DataFrame, key_pairs: dict):
    """
    Filters input dataframe based on the names of the columns and values within

    :param input_data: input dataframe
    :param key_pairs: keys are column names and values are within them
    :return: filtered dataframe or empty
    """
    output = copy.deepcopy(input_data)
    for key, value in key_pairs.items():
        if output.empty:
            return output
        output = filter_by_nan(input_data=output, column_name=key, filter_value=value)
    return output


def get_table_date_range(input_data: pandas.DataFrame,
                         from_column: str | tuple,
                         to_column: str | tuple = None,
                         end_time_moment: str | datetime = None,
                         start_time_moment: str | datetime = None,
                         data_time_period: str | timedelta = None):
    """
    Gets date range from the table

    :param input_data: input data
    :param from_column: main time column
    :param to_column: specify this if data is from to
    :param end_time_moment: time moment to be checked
    :param start_time_moment: start time moment to be checked
    :param data_time_period: time interval
    :return: found start and end time moment from data table
    """
    if isinstance(data_time_period, str):
        data_time_period = parse_duration(data_time_period)
    if not end_time_moment:
        end_time_moment = datetime.now(pytz.utc)
    else:
        end_time_moment = convert_string_to_datetime(end_time_moment)
    if not start_time_moment:
        start_time_moment = end_time_moment - data_time_period
    else:
        start_time_moment = convert_string_to_datetime(start_time_moment)
    if not to_column:
        to_column = from_column
    table_start = (input_data.reset_index())[from_column].min()
    table_end = (input_data.reset_index())[to_column].max()
    table_start = table_start.iloc[0] if isinstance(table_start, pandas.Series) else table_start
    table_end = table_end.iloc[0] if isinstance(table_end, pandas.Series) else table_end

    start_time_moment = max(pandas.to_datetime(table_start), pandas.Timestamp(start_time_moment))
    end_time_moment = min(pandas.to_datetime(table_end), pandas.Timestamp(end_time_moment))
    return  start_time_moment, end_time_moment


def add_levels_to_index(input_df, no_of_levels: int = 0):
    """
    Generates additional empty levels to dataframe

    :param input_df: input dataframe
    :param no_of_levels: number of levels to create
    :return: updated dataframe
    """
    if no_of_levels > 0:
        level_array = [(*c, *[''] * no_of_levels) for c in input_df.columns]
        name_array = input_df.columns.names + [f'level_{x}' for x in range(no_of_levels)]
        input_df.columns = pandas.MultiIndex.from_tuples(tuples=level_array,names=name_array)
    return input_df


def align_two_dataframe_levels(input_df_1: pandas.DataFrame, input_df_2: pandas.DataFrame):
    """
    Sets dataframes to the highest level for merging dataframes with multilevel indices

    :param input_df_1: first dataframe
    :param input_df_2: second dataframe
    :return: updated dataframes
    """
    level_1 = len(input_df_1.columns.levels)
    level_2 = len(input_df_2.columns.levels)
    level_diff = level_2 - level_1
    if level_diff < 0:
        input_df_2 = add_levels_to_index(input_df=input_df_2, no_of_levels = abs(level_diff))
    elif level_diff > 0:
        input_df_1 = add_levels_to_index(input_df=input_df_1, no_of_levels = abs(level_diff))
    return input_df_1, input_df_2


def remove_empty_levels_from_index(input_df):
    """
    Removes empty levels from dataframe

    :param input_df: input dataframe
    :return: updated dataframe
    """
    empty_levels = [x for x in range(input_df.columns.nlevels)
                    if all([y == '' for y in input_df.columns.get_level_values(x)])]
    if empty_levels:
        input_df.columns = input_df.columns.droplevel(empty_levels)
    return input_df


def rename_multilevel_index_levels(input_df, mapping: {}):
    """
    Renames level names for multilevel dataframe (cosmetic puproses only)

    :param input_df: input dataframe
    :param mapping: mapping for the names
    :return: updated dataframe
    """
    try:
        # new_level_names = [mapping.get(x, x) for x in input_df.columns.names]
        new_level_names = [x.replace(y, z) for x in input_df.columns.names for y, z in mapping.items()]
        input_df.columns = input_df.columns.set_names(new_level_names)
    except AttributeError:
        pass
    return input_df


def resample_by_time(data: pandas.DataFrame,
                     index_columns=None,
                     sampling_time: str | timedelta = 'PT15M',
                     method: str = MEAN_KEYWORD,
                     method_parameters: dict = None):
    """
    Resamples the dataframe to the new time interval (for example 15 minutes)
    Question: can ACEol data be regarded as continuous (value in next timestamp is dependent on the value in
    previous timestamp) meaning that when resampling, the data should be meaned. Or data is discrete (meaning that
    in every next timestamp new value is provided and the value from the previous timestamp is handled) meaning that
    when resampling, the data should be summed.

    :param method: specify method to apply resampler, note that 'interpolate' behaves differently
    :param index_columns: columns where timestamps are stored, note that reshape by time can be done using single
        timestamp column, rest must be dropped and recalculated by using the sampling time
    :param data: input dataframe
    :param sampling_time: new sampling rate
    :param method_parameters: dictionary with additional method parameters
    :return: updated dataframe
    """
    post_processing = False
    if index_columns is None:
        # index_columns = [FROM_KEYWORD, TO_KEYWORD]
        index_columns = get_datetime_columns_of_data_frame(input_data=data)
    existing_columns = [column_name for column_name in index_columns if column_name in data.columns.to_list()]
    old_index_columns = []
    # If any of the columns is missing try to reset index
    if len(existing_columns) < len(index_columns):
        old_index_columns = data.index.names
        data = data.reset_index()
        existing_columns = [column_name for column_name in index_columns if column_name in data.columns.to_list()]
    if len(existing_columns) == 0:
        raise ValueError("Cannot resample, no timestamp columns taken")
    if len(existing_columns) > 2:
        raise ValueError("Too many timestamp columns")
    data = str_to_datetime(data=data, columns=existing_columns)
    min_range = {column: min(data[column]) for column in existing_columns}
    # start_time_column = existing_columns[0]
    start_time_column = min(min_range, key=min_range.get)
    existing_diff = get_time_period_from_dataframe_column(input_dataframe=data, data_column=start_time_column)
    new_sampling = parse_duration(sampling_time)
    interpolate = new_sampling <= existing_diff
    end_time_column = None
    if len(existing_columns) == 2:
        # end_time_column = existing_columns[1]
        max_range = {column: max(data[column]) for column in existing_columns}
        end_time_column = max(max_range, key=max_range.get)
        if interpolate:
            last_rows = pandas.DataFrame(data.iloc[[-1]])
            data = pandas.concat([data, last_rows], ignore_index=True)
            data.at[data.index[-1], start_time_column] = data.at[data.index[-1], end_time_column]
        data = data.drop(end_time_column, axis=1)
    if len(existing_columns) > 0:
        post_processing = True
        data = data.drop_duplicates(subset=[start_time_column], keep='last')
        data = data.set_index(start_time_column)
    resampled_data = data.resample(new_sampling)
    if hasattr(resampled_data, method):
        if isinstance(method_parameters, dict) and method_parameters != {}:
            resampled_data = getattr(resampled_data, method)(**method_parameters)
        else:
            resampled_data = getattr(resampled_data, method)()
    else:
        return data
    if post_processing:
        existing_index = start_time_column
        if isinstance(existing_index, str):
            resampled_data = resampled_data.reset_index(names=start_time_column)
        else:
            resampled_data = resampled_data.reset_index()
        if len(existing_columns) == 2:
            if interpolate:
                resampled_data = resampled_data.drop(index=resampled_data.index[-1])
            resampled_data[end_time_column] = resampled_data[start_time_column] + pandas.Timedelta(new_sampling)
    if old_index_columns:
        resampled_data = resampled_data.set_index(old_index_columns)
    return resampled_data
