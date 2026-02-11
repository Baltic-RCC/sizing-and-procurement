import logging
import re
import zoneinfo
from datetime import time, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import aniso8601
import isodate
import pandas
import pytz
from aniso8601 import parse_duration as duration_parser
from dateutil import parser
from dateutil.parser import parse

MIL_TIME_ZONES = {
    'Z': 'Etc/GMT+0',
    'A': 'Etc/GMT+1',
    'B': 'Etc/GMT+2',
    'C': 'Etc/GMT+3',
    'D': 'Etc/GMT+4',
    'E': 'Etc/GMT+5',
    'F': 'Etc/GMT+6',
    'H': 'Etc/GMT+7',
    'G': 'Etc/GMT+8',
    'I': 'Etc/GMT+9',
    'K': 'Etc/GMT+10',
    'L': 'Etc/GMT+11',
    'M': 'Etc/GMT+12',
    'N': 'Etc/GMT-1',
    'O': 'Etc/GMT-2',
    'P': 'Etc/GMT-3',
    'Q': 'Etc/GMT-4',
    'R': 'Etc/GMT-5',
    'S': 'Etc/GMT-6',
    'T': 'Etc/GMT-7',
    'U': 'Etc/GMT-8',
    'V': 'Etc/GMT-9',
    'W': 'Etc/GMT-10',
    'X': 'Etc/GMT-11',
    'Y': 'Etc/GMT-12',
}
TRANSPARENCY_DATETIME_FORMAT = '%Y-%m-%dT%H:%M'
ENTSOE_DATETIME_FORMAT = '%Y%m%d%H%M'

logger = logging.getLogger(__name__)


def extract_timezone_offset(date_time_string: str):
    """
    Additional function to extract timezone info from string for manual conversion

    Note that 2 types are currently searched 1) -02:00 or 2) 'C' or 'CEST'
    NB! Mind the daylight saving

    :param date_time_string: input string
    :return: timezone string if found
    """
    match = re.search(r'([+-]\d{2}:?\d{2}|[A-Z]+$)', date_time_string)
    if match:
        return match.group(1)
    else:
        return None


def parse_datetime_aniso8601(iso_string, keep_timezone=True):
    """
    Wrapper for aniso8601 parse_datetime to keep or discard timezone

    :param iso_string: input string
    :param keep_timezone: whether to keep timezone
    :return: date
    """
    if keep_timezone:
        return aniso8601.parse_datetime(iso_string)
    else:
        return aniso8601.parse_datetime(iso_string).replace(tzinfo=None)


def parse_datetime(iso_string, keep_timezone=True):
    """
    Wrapper for aniso8601 parse_datetime to keep or discard timezone

    :param iso_string: input string
    :param keep_timezone: whether to keep timezone
    :return: datetime object
    """
    try:
        return parse_datetime_aniso8601(iso_string, keep_timezone=keep_timezone)
    except aniso8601.exceptions.ISOFormatError:
        timezone_str = extract_timezone_offset(iso_string)
        tz_infos = None
        if timezone_str is not None:
            time_zone_value = parse_timezone(timezone_str)
            tz_infos = {timezone_str: time_zone_value}
        output_value = parser.parse(iso_string, tzinfos=tz_infos)
        if not keep_timezone:
            return output_value.replace(tzinfo=None)
        return output_value


def convert_string_to_time(input_value: str | time):
    """
    Converts input to datetime object

    :param input_value: input value
    :return:  datetime object
    """
    if not isinstance(input_value, time):
        input_value = aniso8601.time.parse_time(input_value)
    return input_value


def convert_string_to_datetime(input_value: str | datetime):
    """
    Converts input to datetime object

    :param input_value: input value
    :return:  datetime object
    """
    if not isinstance(input_value, datetime):
        input_value = parse_datetime(input_value)
    return input_value


def convert_datetime_to_string(input_value: str | datetime,
                               time_zone: str = 'UTC',
                               output_format: str = TRANSPARENCY_DATETIME_FORMAT):
    """
    Tries to convert input to specified date format string that is needed by transparency platform

    :param time_zone:
    :param input_value: string or datetime object
    :param output_format: format specified for transparency platform
    :return: datetime string
    """
    if isinstance(input_value, str):
        input_value = parse_datetime(input_value)
    if not isinstance(input_value, datetime):
        return input_value
    if time_zone:
        input_value = set_timezone(input_value, time_zone)
    output_value = input_value.strftime(output_format)
    return output_value


def parse_timezone(input_timezone: str = None, return_utc: bool = True):
    """
    Converts string to pytz time zone if applicable

    :param input_timezone: time zone string
    :param return_utc: if fails return utc
    :return: pytz timezone
    """
    time_zone = None
    if return_utc:
        time_zone = pytz.utc
    if input_timezone:
        if not isinstance(input_timezone, str):
            return input_timezone
        try:
            time_zone = ZoneInfo(input_timezone)
        except zoneinfo.ZoneInfoNotFoundError as ex:
            if str(input_timezone).upper() in MIL_TIME_ZONES.keys():
                return ZoneInfo(MIL_TIME_ZONES.get(str(input_timezone).upper()))
            else:
                logger.warning(f"Unknown timezone format: {input_timezone}, got {ex}")
    return time_zone


def set_timezone(input_datetime: str| datetime, time_zone):
    """
    Changes timezone of input if it is applicable, returns input intact if not

    :param input_datetime: string or datetime instance
    :param time_zone: given timezone
    :return: datetime instance
    """
    input_datetime = convert_string_to_datetime(input_value=input_datetime)
    if isinstance(time_zone, str):
        time_zone = parse_timezone(time_zone, return_utc=False)
    if time_zone:
        try:
            input_datetime = input_datetime.astimezone(time_zone)
        except TypeError:
            input_datetime = input_datetime.tz_localize('UTC').astimezone(time_zone)
    return input_datetime


def convert_datetime_to_string_utc(input_value: str | datetime, output_format: str = TRANSPARENCY_DATETIME_FORMAT):
    """
    Tries to convert input to specified date format string to utc

    :param input_value: string or datetime object
    :param output_format: format specified
    :return: datetime string
    """
    if not isinstance(input_value, str) and not isinstance(input_value, datetime):
        return input_value
    return f"{convert_datetime_to_string(input_value=input_value,time_zone='UTC', output_format=output_format)}Z"


def parse_duration(iso8601_duration_string):
    """
    Parses an ISO 8601 duration string and returns the corresponding timedelta object.

    The duration string should be in the format 'PnYnMnDTnHnMnS', where:
    - 'P' is a mandatory prefix indicating the start of the duration
    - 'nY' indicates the number of years in the duration
    - 'nM' indicates the number of months in the duration
    - 'nD' indicates the number of days in the duration
    - 'T' is an optional separator indicating the start of the time portion of the duration
    - 'nH' indicates the number of hours in the duration
    - 'nM' indicates the number of minutes in the duration
    - 'nS' indicates the number of seconds in the duration

    This function allows parsing both positive and negative durations. If the input string
    starts with a '-' sign, the resulting timedelta object will be negated.

    Args:
        iso8601_duration_string (str): The ISO 8601 duration string to parse.

    Returns:
        datetime.timedelta: The timedelta object corresponding to the parsed duration.

    """
    if isinstance(iso8601_duration_string, timedelta):
        return iso8601_duration_string
    try:
        if iso8601_duration_string[0] == "-":
            return duration_parser(iso8601_duration_string[1:]) * -1
        else:
            return duration_parser(iso8601_duration_string)
    except aniso8601.exceptions.ISOFormatError:
        return pandas.Timedelta(iso8601_duration_string)


def str_to_datetime(data: pandas.DataFrame, columns):
    """
    Converts string to pandas date time

    :param data: input dataframe
    :param columns: list of columns to convert
    :return: updated dataframe
    """
    if not isinstance(columns, list):
        columns = [columns]
    for column in columns:
        data.loc[:, column] = pandas.to_datetime(data[column], errors='coerce')
    return data


def get_time_period_from_dataframe_column(input_dataframe: pandas.DataFrame, data_column: str | tuple | list):
    """
    Gets average period between the rows of input dataframe based on the timeseries column

    :param input_dataframe: input data
    :param data_column: column that contains time values
    :return: average period between rows
    """
    if isinstance(data_column, list) and len(data_column) == 2:
        try:
            return (input_dataframe[data_column[0]] - input_dataframe[data_column[1]]).abs().min()
        except (ValueError, Exception):
            pass
    return input_dataframe[data_column].diff().min()


def get_time_intervals(start_date_value: str | datetime,
                       end_date_value: str | datetime,
                       time_delta: str | timedelta = None):
    """
    Calculates time intervals with given time delta between start and end datetime values

    :param start_date_value: start datetime value
    :param end_date_value: end datetime value
    :param time_delta: duration
    :return: list of tuples of start and end dates
    """
    if time_delta is None:
        return [start_date_value, end_date_value]
    time_delta = parse_duration(time_delta)
    start_date_value = convert_string_to_datetime(start_date_value)
    end_date_value = convert_string_to_datetime(end_date_value)
    time_intervals = []
    calculated_start = start_date_value
    calculated_end = min(calculated_start + time_delta, end_date_value)
    while calculated_end < end_date_value:
        time_intervals.append((calculated_start, calculated_end))
        calculated_start = calculated_end
        calculated_end = min(calculated_end + time_delta, end_date_value)
    time_intervals.append((calculated_start, calculated_end))
    return time_intervals


def is_valid_iso8601_duration(input_str: str) -> bool:
    """
    Checks if input string can be converted to duration

    :param input_str: string to be checked
    :return: True if it is valid, False otherwise
    """
    try:
        isodate.parse_duration(input_str)
        return True
    except (isodate.ISO8601Error, TypeError, ValueError):
        return  False


def time_delta_to_str(input_val: Any = None):
    """
    Wraps timedelta conversion to string

    :param input_val: input value, timedelta hopefully
    :return: Parsed timedelta or None
    """
    if not input_val:
        return None
    if isinstance(input_val, str):
        return input_val if is_valid_iso8601_duration(input_str=input_val) else None
    output = isodate.duration_isoformat(input_val)
    if output == 'P%P':
        return None
    return output


def is_datetime_string_column(input_series, sample_size:int =10, threshold=0.8):
    """
    Checks if dataframe column can be parsed to datetime values

    :param input_series: dataframe column
    :param threshold: use this to determine how much values (1->100%) should be able to be converted
    :param sample_size: number of examples
    :return: True if column was able to parse, False otherwise
    """
    input_series = input_series.dropna().astype(str)
    input_series = input_series[~input_series.str.fullmatch(r'\d+')]
    # input_series = input_series[input_series.str.contains(r'[-/:]', regex=True)]
    sample = input_series.dropna().astype(str).head(sample_size)

    counter = 0
    for single_val in sample:
        try:
            parse(single_val, fuzzy=False)
            counter += 1
        except Exception:
            continue
    return (counter / len(sample)) >= threshold if len(sample) > 0 else False


def get_datetime_string_columns_df(input_df, threshold=0.8):
    """
    Checks if dataframe columns can be parsed to datetime values

    :param input_df: dataframe to check
    :param threshold: use this to determine how much values (1->100%) should be able to be converted
    :return: list of columns that can be converted
    """
    return [col_name for col_name in input_df.columns
            if input_df[col_name].dtype == 'object' and
            is_datetime_string_column(input_series=input_df[col_name], threshold=threshold)]


def get_datetime_columns_of_data_frame(input_data: pandas.DataFrame, threshold=0.8):
    """
    Gets column names that are datetime format

    :param input_data: input dataframe
    :param threshold: use this to determine how much values (1->100%) should be able to be converted
    :return: list of columns that are in t
    """
    df_type = input_data.dtypes.to_frame('dtype')
    df_type['dtype_str'] = df_type['dtype'].map(str)
    d_types = df_type[df_type['dtype_str'].str.contains('datetime64')].index.values.tolist()
    if len(d_types) == 0:
        d_types = get_datetime_string_columns_df(input_df=input_data, threshold=threshold)
    return d_types


def set_time_zone(input_data, columns: list | str, time_zone):
    """
    Sets timezone to columns which are specified or which contain datetime objects

    :param input_data: input dataframe
    :param columns: columns to be set
    :param time_zone: desired time zone
    :return: updated dataframe
    """
    if columns is None:
        columns = get_datetime_columns_of_data_frame(input_data=input_data)
    columns = [columns] if not isinstance(columns, list) else columns
    for column_name in columns:
        try:
            try:
                input_data[column_name] = pandas.to_datetime(input_data[column_name]).dt.tz_localize(tz=time_zone)
            except TypeError:
                input_data[column_name] = pandas.to_datetime(input_data[column_name]).dt.tz_convert(tz=time_zone)
        except ValueError:
            input_data[column_name] = input_data[column_name].apply(lambda x: set_timezone(input_datetime=x,
                                                                                           time_zone=time_zone))
        except Exception as ex:
            logger.warning(f"Unable to parse datetime: {ex}")
    return input_data


def check_and_parse_duration(input_str: str = None):
    """
    Checks if string is not None before parsing

    :param input_str: input string
    :return: parsed output or None
    """
    return parse_duration(input_str) if (isinstance(input_str, str) and input_str.strip()) else input_str
