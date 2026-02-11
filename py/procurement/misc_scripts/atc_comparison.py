import copy
import logging
import sys
from io import BytesIO
from datetime import timedelta, datetime

import pandas

from py.common.ref_constants import VALID_FROM_KEY, VALID_TO_KEY
from py.data_classes.task_classes import ProcurementCalculationTask
from py.procurement.constants import PY_DATA_PERIOD_TIME_DELTA, PY_ATC_QUERY, \
    ENTSOE_PIVOT_VALUE, ENTSOE_INDEX_COLUMNS, ATC_COLUMN_NAMES, PY_ATC_FILTER
from py.handlers.elastic_handler import PY_PROCUREMENT_ATC_INDEX, dict_to_and_or_query, get_data_from_elastic_by_time
from py.procurement.procurement_common import get_task_from_environment, align_resolutions, apply_group_by_filter
from py.handlers.minio_handler import save_file_to_minio_with_link

from py.common.functions import save_bytes_io_to_local, rename_multi_index
from py.common.df_functions import slice_data_by_time_range, get_table_date_range
from py.common.time_functions import convert_string_to_datetime, set_timezone, parse_duration


def get_atc_data(atc_query: dict | list = PY_ATC_QUERY,
                 data_period_start_time: str | datetime = None,
                 data_period_end_time: str | datetime = None,
                 atc_index: str = PY_PROCUREMENT_ATC_INDEX,
                 atc_from_column: str = VALID_FROM_KEY,
                 atc_to_column: str = VALID_TO_KEY,
                 group_by_filter: list | dict | str = PY_ATC_FILTER,
                 atc_index_columns: list = None,
                 atc_column_names: list = None,
                 atc_country_area_codes: list = None,
                 atc_pivot_value: list = None,
                 ):
    """
    Downloads and combines atc data

    :param group_by_filter: additional dictionary with column names and functions (basically min max
    :param data_period_start_time: start of calculation
    :param data_period_end_time: end of calculation
    :param atc_index: atc data index
    :param atc_query: query for specify atc
    :param atc_from_column: column name for from data
    :param atc_to_column: column name for to data
    :param atc_pivot_value: value to pivot
    :param atc_index_columns: index for pivot
    :param atc_column_names: in out columns
    :param atc_country_area_codes: list of area codes
    :return:
    """
    atc_data = pandas.DataFrame()
    if not isinstance(atc_query, list):
        atc_query = [atc_query]
    output_data = []
    for single_query in atc_query:
        # As the data is padded forward use some large enough offset towards to the past for getting the values
        # After values are interpolated cut it into requested timeframe
        default_offset = parse_duration('P30D')
        corrected_start_time = convert_string_to_datetime(data_period_start_time) - default_offset
        query_for_data = dict_to_and_or_query(value_dict=single_query, key_name='match')
        data_got = get_data_from_elastic_by_time(start_time_value=corrected_start_time,
                                                 end_time_value=data_period_end_time,
                                                 elastic_index=atc_index,
                                                 elastic_query=query_for_data,
                                                 time_interval_key=atc_from_column)
        data_got[atc_from_column] = data_got[atc_from_column].apply(lambda x: set_timezone(x, 'UTC'))
        data_got[atc_to_column] = data_got[atc_to_column].apply(lambda x: set_timezone(x, 'UTC'))
        # data_got[atc_from_column] = pandas.to_datetime(data_got[atc_from_column]).dt.tz_localize(tz='UTC')
        # data_got[atc_to_column] = pandas.to_datetime(data_got[atc_to_column]).dt.tz_localize(tz='UTC')

        # 2. pivot data
        data_pivot_matches = [x for x in data_got.columns.tolist() for y in atc_pivot_value if
                             y.lower() == x.lower()]
        data_got = apply_group_by_filter(input_data=data_got,
                                         group_by_filter=group_by_filter,
                                         group_by_columns=atc_index_columns + atc_column_names)
        if atc_data.empty:
            atc_data = data_got
        else:
            atc_data = pandas.concat([atc_data, data_got])
        data_pivot = pandas.pivot_table(data_got,
                                       values=data_pivot_matches,
                                       index=atc_index_columns,
                                       columns=atc_column_names,
                                       aggfunc='min')

        output_data.append(data_pivot)
    if len(output_data) == 1:
        atc_pivot = output_data[0]
    else:

        outputs, resolution = align_resolutions(inputs=output_data)
        if len(outputs) == 2:
            day_ahead_df, id_schedules = outputs
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
                for ii in range(len(border_pair)):
                    ahead = border_pair[ii]
                    other = border_pair[ii + 1 if ii < len(border_pair) - 1 else 0]
                    d_plus = next(iter([x for x in c_list if ahead[1] in x[1] and ahead[2] in x[2] and '_1D' in x[0]]))
                    i_minus = next(iter([x for x in c_list if ahead[1] in x[1] and ahead[2] in x[2] and '_ID' in x[0]]))
                    op_plus = next(iter([x for x in c_list if other[1] in x[1] and other[2] in x[2] and '_ID' in x[0]]))
                    merged_atc[ahead] = merged_atc[d_plus] - merged_atc[i_minus] + merged_atc[op_plus]
                    borders.append(ahead)
            atc_pivot = merged_atc[borders]
        else:
            atc_pivot = outputs[0]
    # As offset was applied then cut the data back in to range
    atc_data = slice_data_by_time_range(atc_data, time_ranges=[data_period_start_time, data_period_end_time],
                                        column_to_slice=atc_from_column)
    pivot_from = next(iter([x for x in atc_pivot.reset_index().columns.to_list() if atc_from_column in x]))
    atc_pivot = slice_data_by_time_range(atc_pivot, time_ranges=[data_period_start_time, data_period_end_time],
                                        column_to_slice=pivot_from)
    return atc_data, atc_pivot


def get_atc_data_local(calculation_task: ProcurementCalculationTask = None,
                       data_period_start_time: str | datetime = None,
                       data_period_end_time: str | datetime = None,
                       atc_index: str = PY_PROCUREMENT_ATC_INDEX,
                       atc_query: dict = PY_ATC_QUERY,
                       atc_from_column: str = VALID_FROM_KEY,
                       atc_to_column: str = VALID_TO_KEY,
                       data_period: str | timedelta = PY_DATA_PERIOD_TIME_DELTA,
                       atc_pivot_value: list = None,
                       atc_index_columns: list = None,
                       atc_column_names: list = None,
                       atc_country_area_codes: list= None):

    atc_pivot_value = atc_pivot_value or ENTSOE_PIVOT_VALUE
    if isinstance(atc_pivot_value, str):
        atc_pivot_value = [atc_pivot_value]
    atc_index_columns = atc_index_columns or ENTSOE_INDEX_COLUMNS
    atc_column_names = atc_column_names or ATC_COLUMN_NAMES
    if calculation_task:
        if not atc_country_area_codes:
            atc_country_area_codes = [area.mRID for area in calculation_task.lfc_areas]
        data_period_start_time = data_period_start_time or calculation_task.data_period_start
        data_period_end_time = data_period_end_time or calculation_task.data_period_end
    # 1 get data
    query_for_data = dict_to_and_or_query(value_dict=atc_query, key_name='match')
    atc_data = get_data_from_elastic_by_time(start_time_value=data_period_start_time,
                                             end_time_value=data_period_end_time,
                                             elastic_index=atc_index,
                                             elastic_query=query_for_data,
                                             time_interval_key =atc_from_column)
    if atc_data.empty:
        return atc_data, data_period_start_time, data_period_end_time
    atc_data[atc_from_column] = pandas.to_datetime(atc_data[atc_from_column]).dt.tz_localize(tz='UTC')
    atc_data[atc_to_column] = pandas.to_datetime(atc_data[atc_to_column]).dt.tz_localize(tz='UTC')
    # 2. pivot data
    atc_pivot_matches = [x for x in atc_data.columns.tolist() for y in atc_pivot_value if y.lower() == x.lower()]
    atc_pivot = pandas.pivot_table(atc_data,
                                   values=atc_pivot_matches,
                                   index=atc_index_columns,
                                   columns=atc_column_names,
                                   aggfunc='min')
    pivot_index = atc_pivot.reset_index().columns.tolist()
    new_atc_from_column = [x for x in pivot_index if atc_from_column in x]
    new_atc_to_column = [x for x in pivot_index if atc_to_column in x]
    new_atc_from_column = new_atc_from_column[0] if len(new_atc_from_column) == 1 else atc_from_column
    new_atc_to_column = new_atc_to_column[0] if len(new_atc_to_column) == 1 else new_atc_to_column
    pivot_list = atc_pivot.columns.to_list()
    # 3.filter columns
    if atc_country_area_codes:
        atc_columns = [column for column in pivot_list if len([x for x in column if x in atc_country_area_codes]) >= 2]
        atc_pivot = atc_pivot[atc_columns]
    data_period_start_time, data_period_end_time = get_table_date_range(input_data=atc_pivot,
                                                                        from_column=new_atc_from_column,
                                                                        to_column=new_atc_to_column,
                                                                        start_time_moment=data_period_start_time,
                                                                        end_time_moment=data_period_end_time,
                                                                        data_time_period=data_period)
    return atc_pivot, data_period_start_time, data_period_end_time


if __name__ == '__main__':

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    transparency_atc_data = {'TimeSeries.businessType': 'A31', 'TimeSeries.auction.type': 'A01', 'TimeSeries.contract_MarketAgreement.type': 'A07'}
    ccc_atc_data = {'process.processType': 'A40', 'TimeSeries.businessType': 'A26', 'type': 'A31', 'docStatus.value': 'A37'}
    task = get_task_from_environment()
    c_data, _, _ = get_atc_data_local(calculation_task=task, atc_query=ccc_atc_data)
    rename_values = {area.mRID: f"{area.mRID} ({area.name})" for area in task.lfc_areas}
    rename_values['quantity'] = 'CCC ATC'
    c_rename = {column_name: (rename_values[column_name[0]],
                              f"out_domain {rename_values[column_name[1]]}",
                              f"in_domain {rename_values[column_name[2]]}") for column_name in c_data.columns.to_list()}
    c_data.columns = rename_multi_index(c_data.columns, c_rename)
    merged_data = c_data if not c_data.empty else pandas.DataFrame()
    transparency_data = {}
    t_rename = {}
    for i in range(3):
        transparency_atc_data['TimeSeries.classificationSequence_AttributeInstanceComponent.position'] = str(i + 1)
        t_data, _, _ = get_atc_data_local(calculation_task=task,atc_query=transparency_atc_data)
        rename_values['Point.quantity'] = f'Transparency ATC-sequence {i + 1}'
        t_rename = {column_name: (rename_values[column_name[0]],
                                  f"out_domain {rename_values[column_name[1]]}",
                                  f"in_domain {rename_values[column_name[2]]}") for column_name in
                    t_data.columns.to_list()}
        t_data.columns = rename_multi_index(t_data.columns, t_rename)
        transparency_data[i] = t_data
    merged_transparency = pandas.DataFrame()
    for value in transparency_data.values():
        if merged_transparency.empty:
            merged_transparency = value
        else:
            merged_transparency = merged_transparency.merge(value, left_index=True, right_index=True, how='outer')
    reduced_transparency = copy.deepcopy(merged_transparency)
    column_list = reduced_transparency.columns.tolist()
    for column_name in t_rename.values():
        similar_columns = [x_name for x_name in column_list if x_name[1] == column_name[1] and x_name[2] == column_name[2]]
        new_value_name = ('Transparency ATC value', column_name[1], column_name[2])
        reduced_transparency[new_value_name] = reduced_transparency[similar_columns].min(axis=1, numeric_only=True)
        new_seq_name = ('Transparency ATC sequence',  column_name[1], column_name[2])
        reduced_transparency[new_seq_name] = reduced_transparency[similar_columns].idxmin(axis=1, numeric_only=True)
        reduced_transparency[new_seq_name] = reduced_transparency[new_seq_name].apply(lambda x: str(x[0])[-1])
    reduced_transparency = reduced_transparency.drop(columns=column_list)

    merged_data = merged_data.merge(merged_transparency, left_index=True, right_index=True, how='outer') \
        if not merged_data.empty else merged_transparency
    # merged_data = merged_data.reset_index()
    reduced_data = merged_data.merge(reduced_transparency, left_index=True, right_index=True, how='outer') \
        if not merged_data.empty else reduced_transparency
    # reduced_data = reduced_data.reset_index()
    _, tadas_approach = get_atc_data(atc_query = PY_ATC_QUERY,
                                     data_period_start_time=task.data_period_start,
                                     data_period_end_time= task.data_period_end,
                                     atc_index_columns= ENTSOE_INDEX_COLUMNS,
                                     atc_column_names=ATC_COLUMN_NAMES,
                                     atc_country_area_codes=[area.mRID for area in task.lfc_areas],
                                     atc_pivot_value= ENTSOE_PIVOT_VALUE)
    tadas_column_list = tadas_approach.reset_index().columns.tolist()
    tadas_from = next(iter([x for x in tadas_column_list if VALID_FROM_KEY in x]))
    tadas_to = next(iter([x for x in tadas_column_list if VALID_TO_KEY in x]))
    tadas_approach = tadas_approach.reset_index().set_index([tadas_from, tadas_to])
    tadas_columns = [column for column in tadas_column_list
                     if len([x for x in column if x in [area.mRID for area in task.lfc_areas]]) >= 2]
    tadas_approach = tadas_approach[tadas_columns]
    rename_values['Point.quantity'] = 'ATC proposed by LT'
    d_rename = {column_name: (rename_values[column_name[0]],
                              f"out_domain {rename_values[column_name[1]]}",
                              f"in_domain {rename_values[column_name[2]]}") for column_name in
                tadas_approach.columns.to_list()}
    tadas_approach.columns = rename_multi_index(tadas_approach.columns, d_rename)
    common_keys = [x for x in merged_data.reset_index().columns.to_list()
                   if x in reduced_data.reset_index().columns.to_list() and
                   x in tadas_approach.reset_index().columns.to_list()]
    merged_data = merged_data.reset_index().set_index(common_keys)
    reduced_data = reduced_data.reset_index().set_index(common_keys)
    tadas_approach = tadas_approach.reset_index().set_index(common_keys)
    merged_data = merged_data.merge(tadas_approach, left_index=True, right_index=True, how='outer')
    reduced_data = reduced_data.merge(tadas_approach, left_index=True, right_index=True, how='outer')
    merged_data = merged_data.reset_index()
    reduced_data = reduced_data.reset_index()

    time_cols = {('utc_start', '', ''): ('start time (CET)', '', ''), ('utc_end', '', ''): ('end time (CET)', '', '')}
    for time_column in time_cols.keys():
        merged_data[time_column] = merged_data[time_column].dt.tz_convert(tz='CET')
        merged_data[time_column] = merged_data[time_column].dt.tz_localize(None)
        reduced_data[time_column] = reduced_data[time_column].dt.tz_convert(tz='CET')
        reduced_data[time_column] = reduced_data[time_column].dt.tz_localize(None)
    merged_data.columns = rename_multi_index(merged_data.columns, time_cols)
    reduced_data.columns = rename_multi_index(reduced_data.columns, time_cols)

    folder_to_store = r"E:\margus.ratsep\sizing_of_reserves\reports"
    time_moment_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_file_name = f"ATC_comparison_{time_moment_now}.xlsx"

    print(f"Saving {full_file_name}")
    excel_file = BytesIO()
    with pandas.ExcelWriter(excel_file) as writer:
        merged_data.to_excel(writer, sheet_name='ATC comparison')
        reduced_data.to_excel(writer, sheet_name='ATC min')

    excel_file.name = full_file_name
    to_minio = True
    to_local = True
    if to_minio:
        save_file_to_minio_with_link(excel_file)
    if to_local:
        save_bytes_io_to_local(bytes_io_object=excel_file, location_name=folder_to_store)

    print("Done")
