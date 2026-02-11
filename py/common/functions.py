import ast
import dataclasses
import json
import logging
import os
from datetime import datetime
from io import BytesIO
from itertools import product
from random import choices
from string import ascii_letters

import pandas
from minio.datatypes import JSONDecodeError

from py.common.time_functions import convert_string_to_datetime, parse_timezone, parse_duration
from py.data_classes.enums import ValueOfEnum

logger = logging.getLogger(__name__)


def convert_string_to_expression(input_value):
    """
    Tries to convert input value to python expression

    :param input_value: value to be converted
    :return: expression if successful or input value
    """
    output_value = None
    if isinstance(input_value, str):
        try:
            output_value = ast.literal_eval(input_value)
        except ValueError:
            pass
        except SyntaxError:
            pass
            # logger.warning(f"Syntax error when parsing {input_value}")
    return output_value or input_value


def convert_input(input_value, recursion_depth: int = 3, recursion_counter: int = 1):
    """
    Converts string to list or dict if possible

    :param input_value: input string
    :param recursion_depth:
    :param recursion_counter:
    :return: list or dict if it was possible, None otherwise
    """
    input_value = escape_empty_or_none(input_value)
    if not input_value:
        return None
    output_value = convert_string_to_expression(input_value)
    if isinstance(output_value, str) and recursion_counter < recursion_depth:
        return convert_input(input_value=output_value,
                            recursion_depth=recursion_depth,
                            recursion_counter=recursion_counter + 1)
    return output_value


def load_string_to_list_of_float(input_string):
    """
    Converts string presentation of list to float

    :param input_string: input string
    :return: list of float(s)
    """
    input_string = escape_empty_or_none(input_string)
    if not input_string:
        return None
    list_of_inputs = json.loads(input_string)
    if isinstance(list_of_inputs, list):
        return [float(x) for x in list_of_inputs]
    return [float(list_of_inputs)]


def escape_empty_or_none(input_parameter):
    """
    When parsing strings from config escape empty strings or none keywords

    :param input_parameter: parameter to be escaped
    :return: updated parameter
    """
    if isinstance(input_parameter, str):
        if input_parameter.lower() == 'none' or input_parameter == '':
            return None
    return input_parameter


def parse_to_type(input_parameter, input_type):
    """
    Parses input to given primitive type

    :param input_parameter: parameter to be parsed
    :param input_type: primitive type
    :return: parsed input
    """
    primitives = (bool, str, int, float, type(None))
    escaped_input = escape_empty_or_none(input_parameter)
    if escaped_input and input_type in primitives:
        return input_type(escaped_input)
    return None


def str_to_bool(input_str):
    """
    Tries to convert input to bool

    :param input_str: input string
    :return: bool if was possible, None otherwise
    """
    try:
        return json.loads(str(input_str).lower())
    except JSONDecodeError:
        return None


def calculate_start_and_end_date(start_date_time: str = None,
                                 end_date_time: str = None,
                                 offset: str = None,
                                 time_delta: str = None,
                                 time_zone: str = None,
                                 default_timedelta: str = 'P1D'):
    """
    If no start or end time are provided calculates them or if needed adjusts them by:
    subtracting offset from end time (moving to past)
    adjusting start time by timedelta from end time

    :param time_zone:
    :param start_date_time: start time from when to collect data
    :param end_date_time: end time to when to collect data
    :param offset: value to move end time to past
    :param time_delta: value to adjust start time from end time
    :param default_timedelta: if nothing is specified then subtracts this from end_date_time
    :return: updated start time and end time
    """
    start_date_time = escape_empty_or_none(start_date_time)
    end_date_time = escape_empty_or_none(end_date_time)
    offset = escape_empty_or_none(offset)
    time_delta = escape_empty_or_none(time_delta)
    pytz_timezone = parse_timezone(time_zone)
    if not end_date_time:
        end_date_time = datetime.now(pytz_timezone)
        end_date_time = end_date_time.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        end_date_time = convert_string_to_datetime(end_date_time)
    if offset:
        offset_actual = parse_duration(offset)
        end_date_time = end_date_time - offset_actual
    if time_delta:
        timedelta_actual = parse_duration(time_delta)
        start_date_time = end_date_time - timedelta_actual
    if not start_date_time:
        default_timedelta_actual = parse_duration(default_timedelta)
        start_date_time = end_date_time - default_timedelta_actual
    else:
        start_date_time = convert_string_to_datetime(start_date_time)
    return start_date_time, end_date_time


def save_bytes_io_to_local(bytes_io_object: BytesIO, location_name: str):
    """
    Saves BytesIO object to local storage

    :param bytes_io_object: object to be saved
    :param location_name: where to save
    :return: None
    """
    location_name = check_and_create_the_folder_path(location_name)
    _, file_name = os.path.split(bytes_io_object.name)
    object_file_name = location_name.removesuffix('/') + '/' + file_name.removeprefix('/')
    with open(object_file_name, "wb") as to_write:
        to_write.write(bytes_io_object.getbuffer().tobytes())


def check_the_folder_path(folder_path: str, path_separator: str = '/', os_separator: str = '\\'):
    """
    Checks folder path for special characters.

    :param os_separator:
    :param path_separator:
    :param folder_path: input given
    :return: checked folder path
    """
    if not folder_path.endswith(path_separator):
        folder_path = folder_path + path_separator
    double_separator = path_separator + path_separator
    # Escape '//'
    folder_path = folder_path.replace(double_separator, path_separator)
    # Escape '\'
    folder_path = folder_path.replace(os_separator, path_separator)
    return folder_path


def check_and_create_the_folder_path(folder_path: str):
    """
    Checks if folder path doesn't have any excessive special characters and it exists. Creates it if it does not.

    :param folder_path: input given
    :return: checked folder path
    """
    folder_path = check_the_folder_path(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def add_value_to_dict_recursively(dict_value, output_dict: dict, keys: list, counter: int = 0):
    """
    Adds value to nested dictionary

    :param dict_value: value to be added in the dictionary
    :param output_dict: output dictionary
    :param keys: list of keys to the value
    :param counter: counter for tracking the keys
    :return: updated dictionary
    """
    if counter == len(keys) - 1:
        if isinstance(output_dict, dict):
            output_dict[keys[counter]] = dict_value
        return output_dict

    if counter < len(keys) - 1:
        key = keys[counter]
        key_dict = output_dict.get(key, {})
        key_dict = add_value_to_dict_recursively(dict_value=dict_value, output_dict=key_dict, keys=keys,
                                                 counter=counter + 1)
        output_dict[key] = key_dict
    return output_dict


def get_file_path_from_root_by_name(file_name: str, root_folder: str):
    """
    Searches files from root

    :param file_name: name of the file
    :param root_folder: place from where to start searching
    """
    for (dir_path, _, filenames) in os.walk(root_folder):
        for file_name_found in filenames:
            if file_name_found == file_name:
                return os.path.realpath(os.path.join(dir_path, file_name))
    return file_name


def get_file_path_by_folder(file_path: str, root_folder: str = 'py'):
    """
    Searches file from one directory up from the root folder

    :param file_path: file name / path to search
    :param root_folder: folder from where to start to search
    :return: file path
    """
    one_level_up = os.path.join(os.getcwd().split(root_folder)[0], os.pardir)
    full_path = get_file_path_from_root_by_name(file_name=os.path.basename(file_path), root_folder=one_level_up)
    return full_path


def rename_tuple(tuple_value, mapping_dict):
    """
    Gets new matching value to change in tuple

    :param tuple_value: input tuple value
    :param mapping_dict: mapping dictionary (old: new)
    :return: updated value if found none otherwise
    """
    if tuple_value in mapping_dict.keys():
        return mapping_dict[tuple_value]
    return tuple_value


def rename_multi_index(index, mapper, name_list: list = None):
    """
    For renaming the multilevel index

    :param index: input index
    :param name_list: list of level names
    :param mapper: values to be changed  as a dictionary (old: new)
    :return: updated index
    """
    name_list = name_list or list(index.names)
    tuple_list = [rename_tuple(tuple_value, mapper) for tuple_value in index]
    if max([len(x) for x in tuple_list]) != len(name_list):
        name_list = None
    return pandas.MultiIndex.from_tuples(tuples=[rename_tuple(tuple_value, mapper) for tuple_value in index],
                                         names=name_list)


def align_dataframe_column_names(input_dataframe: pandas.DataFrame, key_words: list):
    """
    Aligns dataframe column names to common format based on the key word list

    :param input_dataframe: input data
    :param key_words: list of words to search in column names
    :return: updated dataframe
    """
    change_columns = input_dataframe.columns.to_list()
    new_columns = {}
    for single_column in change_columns:
        new_column_label = []
        matched_keywords = {}
        for i in range(len(single_column)):
            single_column_label = str(list(single_column)[i])
            found_key_words = [keyword for keyword in key_words if keyword.lower() in single_column_label.lower()]
            if found_key_words:
                matched_keywords[i] = found_key_words
            else:
                new_column_label.append(single_column_label)
        if matched_keywords:
            first_key = min(matched_keywords.keys())
            all_values = []
            for single_value in matched_keywords.values():
                all_values.extend(single_value)
            # Drop duplicates
            all_values = list(set(all_values))
            # Order output to input
            all_values = sorted(all_values, key=lambda x: key_words.index(x))
            new_column_label[first_key:first_key] = ['_'.join(all_values)]
        new_columns[single_column] = (tuple(new_column_label))
    input_dataframe.columns = rename_multi_index(input_dataframe.columns, new_columns)
    return input_dataframe


def get_random_string_length(k: int = 6):
    """
    Generate random string with given length

    :param k: number of characters
    :return: generated string
    """
    return ''.join(choices(ascii_letters, k=k))

def is_nested_dict(input_d: dict):
    """
    Checks if input is nested

    :param input_d: Input dictionary
    :return: True if it is nested, False otherwise
    """
    return any(isinstance(value, dict) for value in input_d.values())


def get_nested_attr(obj, attribute_path, default_value=None):
    """
    Gets nested attribute from object or returns default value if not existent

    :param obj: object itself
    :param attribute_path: attribute name or path (separate nested objects with dot)
    :param default_value: value to be returned if attribute not found
    :return: attribute value or default value
    """
    attributes = attribute_path.split('.')
    for single_attribute in attributes:
        if hasattr(obj, single_attribute):
            obj = getattr(obj, single_attribute)
        else:
            return default_value
    return obj


def align_country_names(input_dataframe: pandas.DataFrame, attribute_name: str, area_list: list = None):
    """
    Sets country names in dataframe columns to the ones specified by attribute names

    :param input_dataframe: input dataframe
    :param area_list: list of areas (Domain, or EICArea instances)
    :param attribute_name: attribute name (if not exists then original is returned)
    :return: input dataframe
    """
    if area_list is None or len(area_list) == 0:
        return input_dataframe
    column_list = input_dataframe.columns.to_list()
    new_column_names = {}
    for column_name in column_list:
        column_name_list = [column_name] if isinstance(column_name, str) else column_name
        single_output = []
        for single_name in column_name_list:
            area_match = next(iter(x for x in area_list if x.value_of(single_name)), None)
            if area_match and (new_value := get_nested_attr(area_match, attribute_name)):
                single_output.append(new_value)
            else:
                single_output.append(single_name)
        if len(single_output) == 1:
            new_column_names[column_name] = single_output[0]
        else:
            new_column_names[column_name] = tuple(single_output)
    if len(new_column_names.keys()) > 0:
        input_dataframe.columns = rename_multi_index(input_dataframe.columns, new_column_names)
    return input_dataframe


def dict_to_dataclass(cls, data):
    """
    Converts to dictionary to object by the keys that have fields in the present in class

    :param cls: to what to convert
    :param data: input as dictionary
    :return: object
    """
    valid_fields = {field_name.name for field_name in dataclasses.fields(cls)}
    filtered_data = {key_name: value_name for key_name, value_name in  data.items() if key_name in valid_fields}
    return cls(**filtered_data)


def check_dict_to_dataclass(cls, data):
    """
    Checks if input is dictionary. If it is converts to object

    :param cls: to what to convert
    :param data: input data
    :return: object
    """
    if isinstance(data, dict):
        return dict_to_dataclass(cls, data)
    return data


def filter_dict_by_enum(input_dict: dict, enum_value: ValueOfEnum):
    """
    Checks if input dict has keys for the given enum type. If it has takes only the one corresponding to
    the key given. Passes through all other key-value pairs

    :param input_dict: dictionary to be filtered
    :param enum_value: Enum value as a filter
    :return: filtered dictionary
    """
    output = {}
    enum_type = enum_value.__class__
    for key_item, value_item in input_dict.items():
        parsed_key = None
        try:
            parsed_key = enum_type.value_of(key_item)
        except ValueError:
            pass
        if parsed_key is None:
            output[key_item] = value_item
        else:
            if parsed_key == enum_value:
                if isinstance(value_item, dict):
                    output = {**output, **value_item}
                else:
                    output[key_item] = value_item
    return output


def unpack_dict_to_lists(input_data):
    """
    Unpacks dictionary (key, values as list) to list of dictionaries where each dictionary contains unique combination
    of the values of the input dictionary

    :param input_data: input dictionary
    :return: list of dictionaries
    """
    for k, v in input_data.items():
        input_data[k] = [v] if not isinstance(v, list) else v
    keys = list(input_data.keys())
    values_product = product(*[input_data[k] for k in keys])
    return [dict(zip(unpack_list(keys), unpack_list(values))) for values in values_product]


def key_exists(input_data: dict, target_key):
    """
    Checks if key exists in nested dictionary recursively

    :param input_data: input dictionary
    :param target_key: key to search
    :return: True if found, False otherwise
    """
    if target_key in input_data or target_key in input_data.values():
        return True
    for v_val in input_data.values():
        if isinstance(v_val, dict):
            if key_exists(v_val, target_key):
                return True
    return False


def update_dict_values(input_dict: dict, update_dict: dict = None, add_dict: dict = None):
    """
    Updates substrings in string in dictionary based on update dictionary  (replace)

    :param input_dict: dictionary to be updated
    :param add_dict: fictionary with values to add
    :param update_dict: dictionary with replacements
    :return: updated dictionary
    """
    output = {}
    for k, v in input_dict.items():
        if update_dict:
            for k_up, v_up in update_dict.items():
                v = v.replace(k_up, v_up)
        output[k] = v
    if add_dict:
        output = {**output, **add_dict}
    return output


def unpack_list(input_data):
    """
    Unpacks lists and tuples within list to list

    :param input_data: list of input values
    :return: unpacked list
    """
    output = []
    for elem in input_data:
        if isinstance(elem, list) or isinstance(elem, tuple):
            output.extend(elem)
        else:
            output.append(elem)
    return output


def ordered_sublist(inputs, lookup):
    """
    Checks if items in inputs are in present in lookup with the order they are

    :param inputs: values to check
    :param lookup: where to check
    :return: true if correct sequence detected
    """
    it = iter(lookup)
    try:
        for i in inputs:
            while next(it) != i:
                pass
        outcome = True
    except StopIteration:
        outcome = False
    return outcome


def get_position_from_list(input_value, input_list, error_value = None):
    """
    Tries to find the element position from list, returns error value otherwise

    :param input_value: key to search
    :param input_list: list from where to search
    :param error_value: default value
    :return:
    """
    try:
        index = input_list.index(input_value)
    except ValueError:
        index = error_value
    return index


def get_value_by_position(key_value, key_list, value_list, error_value = None):
    """
    Returns element from position of key from value_list based key_list

    :param key_value: key to search
    :param key_list: list from where to search key
    :param value_list: values list
    :param error_value: default value
    :return:
    """
    position = get_position_from_list(input_value=key_value, input_list=key_list, error_value=error_value)
    output_value = error_value
    if position != error_value:
        try:
            output_value = value_list[position]
        except IndexError:
            pass
    return output_value
