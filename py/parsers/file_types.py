import io
import json
import logging
import zipfile
from io import BytesIO
from typing import Any
from xml.etree import ElementTree

import openpyxl

from py.parsers.xlsx_to_calculation_result import parse_xlsx_to_list_of_dicts
from py.parsers.xml_to_calculation_result import parse_xml_to_list_of_dicts

logger = logging.getLogger(__name__)


def is_json(input_data: bytes | str = None):
    """
    Checks if data is json.

    :param input_data: string or bytes
    :return: True if it is json
    """
    try:
        json.loads(input_data)
        return True
    except (ValueError, json.decoder.JSONDecodeError, TypeError):
        return False


def is_xml(input_data: bytes | str = None):
    """
    Checks if the input is xml

    :param input_data: string or bytes
    :return: True if it is xml
    """
    try:
        ElementTree.fromstring(input_data)
        return True
    except (ElementTree.ParseError, TypeError):
        return False


def is_xlsx(input_data: bytes | str = None):
    """
    Checks if the input is xlsx

    :param input_data: string or bytes
    :return: True if it is xlsx
    """
    try:
        openpyxl.load_workbook(BytesIO(input_data))
        return True
    except ValueError:
        pass
    except Exception as ex:
        logger.warning(f"Unexpected error {ex}")
    return False


def handle_xml(input_data, filter_dict: dict = None, **kwargs) -> Any | None:
    """
    Converts input xml to dataframe

    :param input_data: bytes or string
    :param filter_dict: use key value pairs to filter the types
    :return: list of dictionaries
    """
    try:
        return parse_xml_to_list_of_dicts(input_data=input_data, filter_dict=filter_dict)
    except AttributeError as ae:
        logger.warning(f"Cannot parse xml to point dataframe: {ae}")
    except Exception as ex:
        logger.warning(f"Unknown error occurred when parsing xml {ex}")
    return None


def handle_json(input_data, **kwargs)-> list | None:
    """
    Converts input json to list of dictionaries
    Caution that the normalization currently done in from elastic to rabbit side
    AAlternative would be to parse input to dataframe and follow the same approach as in handle of xml
    (get point, parents, children and filter by types)

    :param input_data:
    :return: list of dicts
    """
    try:
        output = input_data.decode('utf-8').replace("'",'"')
        output = json.loads(output)
        if isinstance(output, list):
            return output
    except Exception as ex:
        logger.warning(f"Unknown error occurred when parsing json: {ex}")
    return None


def handle_xlsx(input_data, filter_dict: dict = None, **kwargs) -> Any | None:
    """
    Converts input xlsx to dataframe

    :param input_data: bytes or string
    :param filter_dict: use key value pairs to filter the types
    :return: list of dictionaries
    """
    try:
        return parse_xlsx_to_list_of_dicts(input_data=input_data, filter_dict=filter_dict)
    except Exception as ex:
        logger.warning(f"Unknown error occurred when parsing Excel: {ex}")
    return None


def parse_input_by_type(input_data, type_dict: dict, caller: str | object = None, **kwargs):
    """
    Main function to parse

    :param input_data:
    :param type_dict:
    :param caller:
    :return:
    """
    for type_check, type_function in type_dict.items():
        if type_check(input_data):
            caller_name = ''
            if caller:
                if isinstance(caller, object):
                    caller_name = f"{caller.__class__.__name__}: "
                elif isinstance(caller, str):
                    caller_name = caller
            logger.debug(f'{caller_name}: "{type_check.__name__}" is True, parsing with "{type_function.__name__}"')
            return type_function(input_data, **kwargs)
    return None


def is_zip_file(input_data: bytes) -> bool:
    """
    Checks whether the input is zip file (starting bytes and read in)

    :param input_data: Input byte array
    :return: True if it is zip, false otherwise
    """
    if input_data is None:
        return False
    if input_data.startswith(b'PK\x03\x04'):
        try:
            with zipfile.ZipFile(io.BytesIO(input_data)) as read_f:
                return read_f.testzip() is None
        except zipfile.BadZipfile:
            return False
    else:
        return False


def read_in_zip(zip_content: str | io.BytesIO | bytes):
    """
    Reads in files from the given zip file

    :param zip_content: path to the zip file (relative or absolute)
    :return: dictionary with file names as keys and file contents as values
    """
    content = []
    if isinstance(zip_content, bytes):
        zip_content = io.BytesIO(zip_content)
    with zipfile.ZipFile(zip_content, 'r') as zip_file:
        for file_name in zip_file.namelist():
            # logger.info(f"Extracting {file_name}")
            file_content = zip_file.read(file_name)
            bytes_io_file = io.BytesIO(file_content)
            if zipfile.is_zipfile(bytes_io_file):
                zip_content = read_in_zip(bytes_io_file)
                content.extend(zip_content)
            else:
                content.append(file_content)
    return content
