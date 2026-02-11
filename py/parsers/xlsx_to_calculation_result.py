from io import BytesIO
from itertools import chain
from typing import Any

import pandas
import logging

from py.common.functions import rename_multi_index, calculate_start_and_end_date
from py.common.df_functions import parse_dataframe_to_nested_dict
from py.data_classes.enums import ProcurementCalculationType, parse_to_enum_by_value
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_bid import CalculationResultBid
from py.data_classes.results.result_functions import handle_dataframe_timezone_excel
from py.data_classes.task_classes import ProcurementCalculationTask, TimeSliceResult, TimeSlice, TIME_KEYS
from py.parsers.json_to_calculation_result import convert_to_calculation_result, handle_json_output, delete_columns
from py.parsers.parser_constants import PY_OUTPUT_FILE_TYPE, prefixes_to_use
from py.parsers.xml_to_calculation_result import ELASTIC_CHECK_DESIRED_TYPE, filter_list, ELASTIC_CHECK_TYPE_INCLUDED, \
    ELASTIC_CHECK_TYPE_EXCLUDED, convert_columns, filter_dataframe
from py.procurement.constants import QUANTILES_INDEX_NAME, CORRECTED_NCPB_KEY
from py.handlers.elastic_handler import PY_PROCUREMENT_PROPOSED_INDEX
from py.procurement.procurement_common import get_task_from_environment

logger = logging.getLogger(__name__)

EXCEL_INCLUDED_SHEETS = ['ATC', 'NCPB', 'data']
EXCEL_EXCLUDED_SHEETS = ['input', 'pivoted']

PERCENTAGE_KEY = ['percentage']


def read_multi_level_dataframe_from_excel(excel_file: pandas.ExcelFile,
                                          sheet_name: str,
                                          index_columns: int = None,
                                          header_rows: int = None):
    """
    Reads (intended) table (with multilevel index) to multilevel dataframe

    :param index_columns: specify number of index columns
    :param excel_file: Excel file instance
    :param sheet_name: sheet name to be read
    :param header_rows: specify number of header rows
    :return: dataframe
    """
    preview = pandas.DataFrame()
    header_list = None
    index_list = None
    if header_rows is None or index_columns is None:
        preview = excel_file.parse(sheet_name=sheet_name, nrows=10, header=None, index=[1])
    if header_rows is None:
        str_r  = preview.apply(lambda x: x.apply(lambda y: isinstance(y, str)).mean(), axis=1).diff()
        na_r = preview.apply(lambda x: x.isna().mean(), axis=1).diff()
        if na_r.min() == -1:
            header_rows = min(str_r.idxmin(), na_r.idxmin())
        else:
            header_rows = pandas.concat([str_r, na_r], axis=1).ffill().sum(axis=1).idxmin()
        header_list = list(range(header_rows))
    if index_columns is None:
        col_check = header_rows - 1 or 0
        sample_row = preview.iloc[col_check]
        index_columns = 0
        for val in sample_row:
            if isinstance(val, str):
                index_columns += 1
            else:
                break
        index_list = list(range(index_columns)) if index_columns > 0 else None
    output_data = excel_file.parse(sheet_name=sheet_name, header=header_list, index_col=index_list).reset_index()
    return output_data


def parse_excel_to_calculation_results(path_to_excel: str,
                                       task: ProcurementCalculationTask = None,
                                       time_keys: list | str = None,
                                       percentage: list | str = None,
                                       included_sheets: str | list = None,
                                       excluded_sheets: str | list = None,
                                       **kwargs):
    """
    Uploads single Excel sheet to given  elastic index.
    Note that table headings (keys) are converted to snake case for the elastic

    :param percentage: Column name under which is the percentage values
    :param time_keys: column names under which are time values
    :param excluded_sheets: names of sheets to include
    :param included_sheets: name of sheets to exclude
    :param task: Use this instance to specify additional parameters needed for CalculationResult
    :param path_to_excel: path to Excel file contains the Excel sheet name
    """
    included_sheets = included_sheets or EXCEL_INCLUDED_SHEETS
    excluded_sheets = excluded_sheets or EXCEL_EXCLUDED_SHEETS
    time_keys = time_keys or TIME_KEYS
    percentage = percentage or PERCENTAGE_KEY
    time_keys = time_keys if isinstance(time_keys, list) else time_keys
    percentage = percentage if isinstance(percentage, list) else percentage

    results = []
    # Read in Excel
    if isinstance(path_to_excel, bytes):
        path_to_excel = BytesIO(path_to_excel)
    full_excel_file = pandas.ExcelFile(path_to_excel)
    # find page that is closest to given page
    excel_sheets = filter_list(input_data=full_excel_file.sheet_names,
                               included=included_sheets,
                               excluded=excluded_sheets)
    task = task or get_task_from_environment()
    areas = list(chain.from_iterable((area.mRID, area.name) for area in task.lfc_areas))
    if not excel_sheets:
        return results
    results = []
    for sheet_name in excel_sheets:
        excel_dataframe = read_multi_level_dataframe_from_excel(excel_file=full_excel_file, sheet_name=sheet_name)
        excel_dataframe = delete_columns(input_dataframe=excel_dataframe, columns_delete=['index'])
        excel_dataframe.dropna(axis=0, how='all', inplace=True)
        excel_dataframe = handle_dataframe_timezone_excel(input_data=excel_dataframe, timezone='UTC', unlocalize=False)
        c_names = excel_dataframe.columns.to_list()
        percentage_col = filter_list(input_data=c_names, included=percentage)
        type_cols = filter_list(input_data=c_names, included=areas)
        time_cols = filter_list(input_data=c_names, included=time_keys)
        type_value = parse_to_enum_by_value(input_value=sheet_name, enum_type=ProcurementCalculationType)
        if len(type_cols) == 0:
            results.extend(convert_to_calculation_result(input_data=excel_dataframe,
                                                         download_type=type_value,
                                                         task=task))
        else:
            if not percentage_col:
                continue
            if QUANTILES_INDEX_NAME not in percentage_col:
                new_label = {old_key: tuple([QUANTILES_INDEX_NAME if str(y).lower() in str(x).lower() else x
                                             for x in old_key for y in percentage]) for old_key in percentage_col}
                excel_dataframe.columns = rename_multi_index(excel_dataframe.columns, new_label)
                percentage_col = list(new_label.values())
            if len(time_cols) > 0:
                valid_from = (excel_dataframe[time_cols].min()).min()
                valid_to = (excel_dataframe[time_cols].max()).max()
            else:
                valid_to, valid_from = calculate_start_and_end_date()
            time_slice = TimeSlice(valid_from=valid_from, valid_to=valid_to)
            type_value = parse_to_enum_by_value(input_value=sheet_name, enum_type=ProcurementCalculationType)
            reduced_df = excel_dataframe[percentage_col + type_cols].set_index(percentage_col)
            time_slice_results = [TimeSliceResult(calculation_type=type_value,
                                                  time_slice = time_slice,
                                                  quantile_result=reduced_df)]
            new_result = None
            corrected = CORRECTED_NCPB_KEY in sheet_name
            if type_value == ProcurementCalculationType.ATC:
                new_result = CalculationResultATC(task=task, pivoted_data=excel_dataframe, quantiles=time_slice_results)
            elif type_value == ProcurementCalculationType.NCPB:
                new_result = CalculationResultBid(task=task,
                                                  pivoted_data=excel_dataframe,
                                                  NCPB_updated=corrected,
                                                  quantiles=time_slice_results)
            if new_result:
                new_result.valid_from = valid_from
                new_result.valid_to = valid_to
                # new_result.set_domains()
                results.append(new_result)
    return results


def parse_xlsx_to_dataframe(input_data, sheet_included_keywords: list = None, sheet_excluded_keywords: list = None):
    """
    Converts excel to dataframe

    :param input_data: Excel file
    :param sheet_included_keywords: keywords for sheet names to include
    :param sheet_excluded_keywords: keywords for sheet names to exclude
    :return: filtered sheet names
    """
    full_excel_file = pandas.ExcelFile(input_data)
    sheets = filter_list(input_data=full_excel_file.sheet_names,
                         included=sheet_included_keywords,
                         excluded=sheet_excluded_keywords)
    outputs = [full_excel_file.parse(sheet_name) for sheet_name in sheets]
    return outputs


def parse_xlsx_to_list_of_dicts(input_data,
                                filter_dict: dict = None,
                                sheet_included_keywords: list = None,
                                sheet_excluded_keywords: list = None,
                                elastic_check_included_keywords: list = None,
                                elastic_check_excluded_keywords: list = None,
                                elastic_check_type: str = ELASTIC_CHECK_DESIRED_TYPE) -> Any | None:
    """
    Converts input xml to dataframe

    :param sheet_excluded_keywords:
    :param sheet_included_keywords:
    :param elastic_check_type: change columns to this data type
    :param elastic_check_excluded_keywords: keywords for columns to change
    :param elastic_check_included_keywords: keywords for columns to exclude from change
    :param input_data: bytes or string
    :param filter_dict: use key value pairs to filter the types
    :return: list of dictionaries
    """
    elastic_check_included_keywords = elastic_check_included_keywords or ELASTIC_CHECK_TYPE_INCLUDED
    elastic_check_excluded_keywords = elastic_check_excluded_keywords or ELASTIC_CHECK_TYPE_EXCLUDED
    logger.info(f"Received xlsx")
    excel_dfs = parse_xlsx_to_dataframe(input_data=input_data,
                                        sheet_included_keywords=sheet_included_keywords,
                                        sheet_excluded_keywords=sheet_excluded_keywords)
    single_df = pandas.concat(excel_dfs)
    output = convert_columns(input_data=single_df,
                             desired_type=elastic_check_type,
                             included=elastic_check_included_keywords,
                             excluded=elastic_check_excluded_keywords)
    if filter_dict and isinstance(filter_dict, dict):
        output = filter_dataframe(input_data=output, key_value_pairs=filter_dict)
    if not output.empty:
        return parse_dataframe_to_nested_dict(input_dataframe=output)
    logger.info(f"Unable to parse dataframe, may be empty")
    return None


if __name__ == '__main__':

    # file_name = r"E:\margus.ratsep\sizing_of_reserves\reports\RCC_Proposed-NCPB-ATC-10-07-2025_08-50-07.xlsx"
    file_name = r"E:\margus.ratsep\sizing_of_reserves\reports\RCC_Proposed-NCPB-ATC-03-08-2025_16-24-03.xlsx"
    environment_task = get_task_from_environment()
    all_results = parse_excel_to_calculation_results(path_to_excel=file_name,
                                                     task=environment_task,
                                                     included_sheets=EXCEL_INCLUDED_SHEETS,
                                                     excluded_sheets=EXCEL_EXCLUDED_SHEETS)
    area_codes = []
    for result in all_results:
        result.update_version_numbers(elastic_index=PY_PROCUREMENT_PROPOSED_INDEX)
    if all_results is not None and len(all_results) > 0:
        handle_json_output(results=all_results, prefixes=prefixes_to_use, output_file_type=PY_OUTPUT_FILE_TYPE)
        # results_to_elastic(results=all_results, elastic_index=PY_PROCUREMENT_PROPOSED_INDEX)
    print("Done")
