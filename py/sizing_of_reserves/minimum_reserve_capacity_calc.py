import logging
import os
import sys
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import config
from py.common.config_parser import parse_app_properties
from py.data_classes.enums import InputColumnName
from py.handlers.minio_handler import save_file_to_minio_with_link
from py.common.ref_constants import MEAN_KEYWORD, \
    INTERPOLATE, INTERPOLATE_PARAMETERS, VALID_FROM_KEY, VALID_TO_KEY, DOMAIN_NAME_KEY, \
    POINT_QUANTITY_KEY
from py.handlers.elastic_handler import ElkHandler, PY_ELASTICSEARCH_HOST, \
    get_data_from_elastic_by_time, PY_SIZING_CURRENT_BALANCING_STATE_INDEX
from py.common.functions import calculate_start_and_end_date, \
    escape_empty_or_none, str_to_bool, convert_input
from py.common.df_functions import slice_data_by_time_range, resample_by_time, get_column_names_from_data
from py.common.time_functions import parse_datetime, parse_timezone, set_time_zone
from py.data_classes.results.result_functions import handle_dataframe_timezone_excel
from py.sizing_of_reserves.sizing_data_classes import CapacityDataSeriesCenterLine, CapacityDataSeries, MinimumCapacityData, \
    FrequencyRestorationControlError, ceil
from py.sizing_of_reserves.report_generation import generate_report_pdf, PLUS_MINUS, ALL_DATA_KEYWORD, \
    POSITIVE_DATA_KEYWORD, NEGATIVE_DATA_KEYWORD, DET_DESCRIPTION_KEYWORD, DETERMINISTIC_FIGURE_NAME, \
    MC_DESCRIPTION_KEYWORD, MC_FIGURE_NAME, DATE_FORMAT_FOR_REPORT, generate_report_word

logger = logging.getLogger(__name__)
parse_app_properties(globals(), config.paths.config.sizing_reserves)

STD_KEYWORD = 'std'
REGIONS = ['Baltics', 'Estonia', 'Latvia', 'Lithuania']
INPUT_DATA_FIGURE_NAME = 'input_data.png'
INITIAL_ALLOWED = 'Max. allowed'
INITIAL_UNCORRECTED = "Actual"
TARGET = 'target'
LOWER_BOUND = 'lower value'
UPPER_BOUND = 'upper value'
EXCESS_WHEN_APPLIED = 'after applying'
PY_TIME_KEY_FROM = escape_empty_or_none(TIME_KEY_FROM) or 'valid_from'
PY_TIME_KEY_TO = escape_empty_or_none(TIME_KEY_TO) or 'valid_to'

IMBALANCE_COLUMN_KEYS = {InputColumnName.START_TIME_COLUMN: [VALID_FROM_KEY, 'start_time'],
                         InputColumnName.END_TIME_COLUMN: [VALID_TO_KEY, 'end_time'],
                         InputColumnName.VALUE_COLUMN: [POINT_QUANTITY_KEY, 'value.value', 'quantity', 'Point.quantity'],
                         InputColumnName.REGION_COLUMN: [DOMAIN_NAME_KEY, 'in_domain.name']}

DEFAULT_INDEX_KEYS = [InputColumnName.START_TIME_COLUMN, InputColumnName.END_TIME_COLUMN]
DEFAULT_IMBALANCE_GROUP_KEYS = [InputColumnName.REGION_COLUMN]
DEFAULT_VALUE_KEY = [InputColumnName.VALUE_COLUMN]


NUMBER_OF_SIMULATIONS = 50
NUMBER_OF_SAMPLES = 105000

# Constants to be loaded in from config file
PY_SIZING_OFFSET = escape_empty_or_none(SIZING_OFFSET)
PY_SIZING_TIME_DELTA = escape_empty_or_none(SIZING_TIME_DELTA)
PY_SIZING_ANALYSIS_DATE = escape_empty_or_none(SIZING_ANALYSIS_DATE)
PY_SIZING_CALCULATION_RESOLUTION = escape_empty_or_none(SIZING_CALCULATION_RESOLUTION)
PY_SIZING_OUTPUT_TO_PDF = str_to_bool(str(SIZING_OUTPUT_TO_PDF).lower())
PY_SIZING_OUTPUT_TO_WORD = str_to_bool(str(SIZING_OUTPUT_TO_WORD).lower())
PY_SIZING_TIMEZONE = SIZING_TIMEZONE
PY_SIZING_PERCENTILES = convert_input(SIZING_PERCENTILES)
ANALYSIS_PERCENTILES = {f"{float(x)}%": round(float(x) / 100, 4) for x in PY_SIZING_PERCENTILES}


def find_percentiles(data: pd.DataFrame,
                     column_name: str,
                     percentiles: {},
                     percentile_values: {}):
    """
    Composes centerlines (CapacityDataSeriesCenterLine) from percentiles

    :param percentile_values: if given and applicable then finds mean of the already calculated percentiles
    :param data: input data
    :param column_name: column from which to calculate
    :param percentiles: percentiles to be added
    :return: list of CapacityDataSeriesCenterLine instances
    """
    centres = []
    for percentile in percentiles:
        deviation_value = 0
        if percentile in percentile_values and MEAN_KEYWORD in percentile_values[percentile]:
            percentile_value = percentile_values[percentile][MEAN_KEYWORD]
            if STD_KEYWORD in percentile_values[percentile]:
                deviation_value = percentile_values[percentile][STD_KEYWORD]
        else:
            percentile_value = data[column_name].quantile(q=percentiles[percentile])
        center = CapacityDataSeriesCenterLine(center_line=percentile_value,
                                              label=percentile,
                                              percentile=percentiles[percentile],
                                              deviation=deviation_value)
        centres.append(center)
    return centres


def compose_data_series(data: pd.DataFrame,
                        column_name: str,
                        label: str,
                        percentiles: {},
                        percentile_values: {} = None,
                        show_percentiles: bool = True):
    """
    Wraps the data to custom class that draws a figure of data and percentiles with values on it

    :param percentile_values: custom percentile values
    :param data: input data, basis of the graph
    :param column_name: column from which the
    :param label: data[column_name] legend entry
    :param percentiles: percentiles to be added to the figure (note that show_percentiles should be True in this case)
    :param show_percentiles: if percentiles would be shown
    :return: custom class
    """
    if percentile_values is None:
        percentile_values = {}
    center_lines = []
    data_title = ""
    all_lines = len(data)
    if show_percentiles:
        center_lines = find_percentiles(data=data,
                                        column_name=column_name,
                                        percentiles=percentiles,
                                        percentile_values=percentile_values)
        if len(center_lines) == 2:
            center_line_values = [x.center_line for x in center_lines]
            min_center_line = min(center_line_values)
            max_center_line = max(center_line_values)
            if min_center_line < data[column_name].mean() < max_center_line:
                data_title = (
                    f"+: {len(data[(data[column_name] > max_center_line)]) * 100 / all_lines:.3f}% / "
                    f"-: {len(data[(data[column_name] < min_center_line)]) * 100 / all_lines:.3f}%")
    return CapacityDataSeries(input_data=data,
                              label=label,
                              title=data_title,
                              column=column_name,
                              center_lines=center_lines)


def run_deterministic_analysis(data: pd.DataFrame,
                               main_column_name: str,
                               percentiles: {},
                               use_pos_neg_data_separately: bool = False,
                               show_percentiles: bool = True,
                               image_name: str = DETERMINISTIC_FIGURE_NAME):
    """
    Runs the deterministic approach based on assumption that data is normally distributed. Finds center lines
    as fixed percentiles. Handles three cases: all data, extracted positive values and extracted negative values

    :param percentiles:
    :param show_percentiles:
    :param image_name:
    :param use_pos_neg_data_separately:
    :param data: dictionary as {region name: region data as pandas dataframe}
    :param main_column_name: name of the region or the column in which to carry out the analysis
    :return: MinimumCapacityData instance
    """
    # ElkHandler.sys_print(f"\rAnalysing FRR data in {main_column_name}")
    logger.info(f"Deterministic analysis. Analysing FRR data in {main_column_name}")
    data_to_image = [compose_data_series(data=data,
                                         column_name=main_column_name,
                                         label=ALL_DATA_KEYWORD,
                                         percentiles=percentiles,
                                         show_percentiles=show_percentiles)]
    # Following part is for illustration purposes only. Do not proceed
    if use_pos_neg_data_separately:
        pos_data = data.loc[data[main_column_name] >= 0]
        neg_data = data.loc[data[main_column_name] <= 0]
        pos_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value >= 0.5}
        neg_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value <= 0.5}
        data_to_image.append(compose_data_series(data=pos_data,
                                                 column_name=main_column_name,
                                                 label=POSITIVE_DATA_KEYWORD,
                                                 percentiles=pos_percentiles,
                                                 show_percentiles=show_percentiles))
        data_to_image.append(compose_data_series(data=neg_data,
                                                 column_name=main_column_name,
                                                 label=NEGATIVE_DATA_KEYWORD,
                                                 percentiles=neg_percentiles,
                                                 show_percentiles=show_percentiles))
    result = MinimumCapacityData(description=DET_DESCRIPTION_KEYWORD,
                                 input_data=data_to_image)
    result.plot_data(image_name)
    logger.info(f"Deterministic analysis done")
    # ElkHandler.sys_print(f"\rAnalysing FRR data in {main_column_name}: Done\n")
    return result


def simulate_mc(data: pd.DataFrame,
                data_column: str,
                percentiles: {},
                number_of_samples: int = NUMBER_OF_SAMPLES,
                number_of_simulations: int = NUMBER_OF_SIMULATIONS):
    """
    Simulates Monte Carlo on data frame (number_of_simulations x number_of_samples). Returns the dataframe consisting
    of percentiles of the requested column retrieved after every simulation

    :param data: input dataframe
    :param data_column: column in which percentiles are found after each simulation
    :param percentiles: dictionary of percentiles {percentile_name: percentile_value}
    :param number_of_samples: number of randomly selected samples per simulation
    :param number_of_simulations: number of simulations
    :return: dataframe consisting of percentile values
    """
    sim_results = []
    for sim in range(number_of_simulations):
        data_mc = data.sample(number_of_samples, replace=True)
        sim_result = {'n': sim}
        for percentile in percentiles:
            sim_result[percentile] = data_mc[data_column].quantile(q=percentiles[percentile])
            sim_results.append(sim_result)
        # ElkHandler.sys_print(f"\rSimulating FRR in {data_column}: "
        #                                 f"{100 * sim / number_of_simulations:.2f}% done")
    return pd.DataFrame(sim_results)


def get_mean_of_columns(data: pd.DataFrame, columns: list, find_std: bool = True):
    """
    Calculates mean to columns indicated (and standard deviation if needed)

    :param data: input data frame
    :param columns: list of columns for which mean is needed
    :param find_std: True: calculates standard deviation also
    :return: dictionary {column_name: {'mean': mean of column, Optional('std': standard deviation of column)}}
    """
    results = {}
    for column in columns:
        if column in data.columns:
            results[column] = {MEAN_KEYWORD: data[column].mean()}
            if find_std:
                results[column][STD_KEYWORD] = data[column].std()
    return results


def run_mc_on_data(data: pd.DataFrame,
                   main_column_name: str,
                   percentiles: {},
                   use_pos_neg_data_separately: bool = True,
                   show_percentiles: bool = True,
                   number_of_samples: int = NUMBER_OF_SAMPLES,
                   number_of_simulations: int = NUMBER_OF_SIMULATIONS):
    """
    Runs 'probabilistic' analysis by sampling randomly input data and extracts given fixed percentiles
    as center lined

    :param percentiles:
    :param use_pos_neg_data_separately:
    :param show_percentiles: show percentiles
    :param data: dictionary as {region name: region data as pandas dataframe}
    :param main_column_name: name of the region
    :param number_of_samples: samples per simulation
    :param number_of_simulations: number of simulations
    :return: MinimumCapacityData instance
    """
    logger.info(f"MC analysis. Simulating FRR in {main_column_name}")
    all_data_simulated = simulate_mc(data=data,
                                     data_column=main_column_name,
                                     percentiles=percentiles,
                                     number_of_samples=number_of_samples,
                                     number_of_simulations=number_of_simulations)
    all_data_mean = get_mean_of_columns(data=all_data_simulated, columns=percentiles.keys())
    data_series = [compose_data_series(data=data,
                                       column_name=main_column_name,
                                       label=ALL_DATA_KEYWORD,
                                       percentiles=percentiles,
                                       percentile_values=all_data_mean,
                                       show_percentiles=show_percentiles)]
    # Following part is for illustration purposes only. Do not proceed
    if use_pos_neg_data_separately:
        pos_data = data.loc[data[main_column_name] >= 0]
        pos_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value >= 0.5}
        pos_data_simulated = simulate_mc(data=pos_data,
                                         data_column=main_column_name,
                                         percentiles=pos_percentiles,
                                         number_of_samples=number_of_samples,
                                         number_of_simulations=number_of_simulations)
        pos_data_mean = get_mean_of_columns(data=pos_data_simulated, columns=list(pos_percentiles.keys()))
        data_series.append(compose_data_series(data=pos_data,
                                               column_name=main_column_name,
                                               label=POSITIVE_DATA_KEYWORD,
                                               percentiles=pos_percentiles,
                                               percentile_values=pos_data_mean,
                                               show_percentiles=show_percentiles))
        neg_data = data.loc[data[main_column_name] <= 0]
        neg_percentiles = {p_key: p_value for p_key, p_value in percentiles.items() if p_value <= 0.5}
        neg_data_simulated = simulate_mc(data=neg_data,
                                         data_column=main_column_name,
                                         percentiles=neg_percentiles,
                                         number_of_samples=number_of_samples,
                                         number_of_simulations=number_of_simulations)
        neg_data_mean = get_mean_of_columns(data=neg_data_simulated, columns=list(neg_percentiles.keys()))
        data_series.append(compose_data_series(data=neg_data,
                                               column_name=main_column_name,
                                               label=NEGATIVE_DATA_KEYWORD,
                                               percentiles=neg_percentiles,
                                               percentile_values=neg_data_mean,
                                               show_percentiles=show_percentiles))
    result = MinimumCapacityData(description=MC_DESCRIPTION_KEYWORD,
                                 input_data=data_series)
    result.plot_data(MC_FIGURE_NAME)
    logger.info(f"MC analysis done")
    # ElkHandler.sys_print(f"\rSimulating FRR in {main_column_name}: Done\n")
    return result


def draw_input_data(data: pd.DataFrame, regions: list, file_name: str = None):
    """
    Draws a figure depicting the input data

    :param regions: list of regions to show
    :param data: dataframe with ACEol data
    :param file_name: name of location where and if to save
    :return: none
    """
    if VALID_FROM_KEY not in data.columns:
        return
    x_min = data[VALID_FROM_KEY].min()
    x_max = data[VALID_FROM_KEY].max()
    fig_cols = int(min(2, len(regions)))
    fig_rows = int(ceil(len(regions) / fig_cols))
    _, axes = plt.subplots(fig_rows, fig_cols, squeeze=False)
    for i, region in enumerate(regions):
        # y_max = max(abs(data[regions[i]])) * 1.2
        image_col = i % fig_cols
        image_row = i // fig_rows
        time_series = data.plot(x=VALID_FROM_KEY, y=region, ax=axes[image_row][image_col], label=region)
        # time_series = data.plot(x=FROM_KEYWORD, y=region)
        time_series.set_title(f"Input imbalance volume data")
        time_series.set_xlim(x_min, x_max)
        time_series.grid()
        time_series.set_xlabel("Time")
        time_series.set_ylabel("Imbalance volume data (MW)")
    if file_name is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')


def delete_file_from_local_storage(file_path: str):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.info(f"Removed {file_path} from local storage")
    else:
        logger.warning(f"Unable to delete {file_path}, file doesn't exist")


def get_imbalance_volume_data(start_time_value: str | datetime,
                              end_time_value: str | datetime,
                              elk_server: str = PY_ELASTICSEARCH_HOST,
                              elastic_index: str = PY_SIZING_CURRENT_BALANCING_STATE_INDEX,
                              aceol_query: dict = None,
                              time_start_key: str | list = PY_TIME_KEY_FROM,
                              time_end_key: str | tuple = PY_TIME_KEY_TO,
                              dict_to_flat: bool = True,
                              use_default_fields: bool = True,
                              mapping: dict = None,
                              index_columns: str | list = None,
                              value_column: str | tuple = None,
                              name_columns: str | tuple | list = None
                              ):
    """
    Requests aceol data and reshapes it to regions (states)

    :param name_columns: columns with labels
    :param value_column: columns from where the values are
    :param index_columns: columns to set index
    :param start_time_value: from where to start to query
    :param end_time_value: to where to query
    :param elk_server: address to the server
    :param elastic_index: index where data is stored
    :param mapping: for translating input columns to common ones
    :param time_start_key: time start key by which to query
    :param time_end_key: time end key
    :param use_default_fields: whether to query fields from elastic
    :param dict_to_flat: whether to parse nested dicts to single dicts
    :param elastic_index: index where is imbalance volumes
    :param aceol_query: dictionary containing query (use dev-tools of ElasticSearch to put it together)
    :return: pandas dataframe with columns to-from and states
    """
    mapping = mapping or IMBALANCE_COLUMN_KEYS
    index_columns= index_columns or DEFAULT_INDEX_KEYS
    name_columns = name_columns or DEFAULT_IMBALANCE_GROUP_KEYS
    value_column = value_column or DEFAULT_VALUE_KEY
    data_aceol = get_data_from_elastic_by_time(start_time_value=start_time_value,
                                               end_time_value=end_time_value,
                                               elk_server=elk_server,
                                               elastic_index=elastic_index,
                                               elastic_query=aceol_query,
                                               time_interval_key=time_start_key,
                                               dict_to_flat=dict_to_flat,
                                               use_default_fields=use_default_fields)
    data_mapping = get_column_names_from_data(input_data=data_aceol, input_mapping=mapping)
    data_aceol = set_time_zone(input_data=data_aceol, columns=[time_start_key, time_end_key], time_zone='UTC')

    index_col = [data_mapping.get(x) for x in index_columns]
    type_col = [data_mapping.get(x) for x in name_columns]
    value_col = [data_mapping.get(x) for x in value_column][0]

    aceol_by_states = pd.pivot_table(data_aceol,
                                     values=value_col,
                                     index=index_col,
                                     columns=type_col)
    aceol_by_states = aceol_by_states.reset_index(names=index_col)
    return aceol_by_states


if __name__ == '__main__':
    """ RUN THIS """

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    region_to_investigate = REGIONS[0]
    draw_raw_data_image = True  # set it to true if image with raw data adds some value, for example estimate
    # scale of fluctuation on the timescale
    pytz_timezone = parse_timezone(PY_SIZING_TIMEZONE)
    try:
        calculation_date = parse_datetime(PY_SIZING_ANALYSIS_DATE)
    except ValueError:
        calculation_date = datetime.now(pytz_timezone)
    end_time = calculation_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time, end_time = calculate_start_and_end_date(end_date_time=end_time,
                                                        time_delta=PY_SIZING_TIME_DELTA,
                                                        time_zone=pytz_timezone,
                                                        offset=PY_SIZING_OFFSET)

    """------------------------Load the data-----------------------------------------------------------------------"""

    elk = ElkHandler(server=PY_ELASTICSEARCH_HOST)
    imbalance_volume_data = get_imbalance_volume_data(start_time_value=start_time, end_time_value=end_time)
    imbalance_volume_data = handle_dataframe_timezone_excel(input_data=imbalance_volume_data,
                                                            timezone='UTC', unlocalize=False)

    # TODO: come up with better approach
    if 'Baltics' not in imbalance_volume_data.columns.to_list():
        imbalance_volume_data['Baltics'] = imbalance_volume_data[['Estonia', 'Latvia', 'Lithuania']].sum(axis=1)
    else:
        imbalance_volume_data['Baltics'] = (imbalance_volume_data['Baltics']
                                            .fillna(imbalance_volume_data[['Estonia', 'Latvia', 'Lithuania']].sum(axis=1)))

    """------------------------Preprocessing-----------------------------------------------------------------------"""
    # Preprocessing: resample it to 15 min (by methodology), and slice a period from 1.5 years to 0.5 from current time
    # stamp
    # imbalance_volume_data = str_to_datetime(imbalance_volume_data, [FROM_KEYWORD, TO_KEYWORD])
    imbalance_volume_data = resample_by_time(imbalance_volume_data,
                                             sampling_time=PY_SIZING_CALCULATION_RESOLUTION,
                                             method=INTERPOLATE,
                                             method_parameters=INTERPOLATE_PARAMETERS)
    # Leave the left endpoint free: AT LEAST one year of data ending not before than 6 months before the analysis
    # time_range[START_DATE_KEYWORD] = None
    imbalance_volume_data = imbalance_volume_data.ffill()
    imbalance_volume_data_in_time_range = slice_data_by_time_range(data=imbalance_volume_data,
                                                                   time_ranges=[start_time, end_time],
                                                                   column_to_slice=VALID_FROM_KEY)
    if not os.path.isfile(INPUT_DATA_FIGURE_NAME):
        if draw_raw_data_image:
            logger.info(f"Saving image with raw and uncompressed data, it may take a while...")
            draw_input_data(imbalance_volume_data_in_time_range, [region_to_investigate], INPUT_DATA_FIGURE_NAME)
    # Update the left endpoint from the data
    start_time = imbalance_volume_data_in_time_range[VALID_FROM_KEY].min()
    index_values = {}
    for x_key, x_value in ANALYSIS_PERCENTILES.items():
        if x_value < 0.5:
            index_values[f"{100 * (1 - x_value)}% for -"] = x_value
        else:
            index_values[f"{x_key} for +"] = x_value

    negative_values = any(x for x in dict(ANALYSIS_PERCENTILES).values() if float(x) < 0)
    over_tuned = any(x for x in dict(ANALYSIS_PERCENTILES).values() if 0 < float(x) < 0.5)
    if negative_values and over_tuned:
        logger.error(f"Both negative and less than 0.5 values exists. Choose one approach")
        sys.exit()

    if negative_values:
        for x_key, x_value in dict(ANALYSIS_PERCENTILES).items():
            if x_value < 0:
                ANALYSIS_PERCENTILES[x_key] = round(1 + x_value, 4)

    """------------------------Run 'main analysis'-----------------------------------------------------------------"""
    # Draw percentiles on the raw data
    # analysis_percentiles = {PERCENTILE_9999: PERCENTILE_9999_VALUE, PERCENTILE_0001: PERCENTILE_0001_VALUE}

    det_data = run_deterministic_analysis(data=imbalance_volume_data_in_time_range,
                                          main_column_name=region_to_investigate,
                                          use_pos_neg_data_separately=False,
                                          percentiles=ANALYSIS_PERCENTILES)
    # Just for the fun of it, resample the normal distribution to get the normal distribution, after which draw the
    # percentiles and conclude that results are same than in previous case
    mc_data = run_mc_on_data(data=imbalance_volume_data_in_time_range,
                             main_column_name=region_to_investigate,
                             use_pos_neg_data_separately=False,
                             percentiles=ANALYSIS_PERCENTILES,
                             number_of_simulations=50,
                             number_of_samples=300000)
    """------------------------Run 'main analysis'-----------------------------------------------------------------"""
    index_values = {}
    for x_key, x_value in ANALYSIS_PERCENTILES.items():
        if x_value < 0.5:
            index_values[f"{100 * (1 - x_value)}% for -"] = x_value
        else:
            index_values[f"{x_key} for +"] = x_value

    # index_values = {'99.99% for +': PERCENTILE_9999_VALUE, '99.99% for -': PERCENTILE_0001_VALUE}
    # Prepare for the report
    det_data_report = det_data.get_report(index_list=index_values)
    mc_data_report = mc_data.get_report(index_list=index_values)
    overall_report = pd.concat([det_data_report, mc_data_report], axis=1)
    # overall_report.to_csv('analysis_results.csv', sep=DELIMITER)

    min_percentile_value = min(ANALYSIS_PERCENTILES.values())
    max_percentile_value = max(ANALYSIS_PERCENTILES.values())
    min_all_value_det = det_data.get_value(index_value=min_percentile_value, data_name=ALL_DATA_KEYWORD)
    max_all_value_det = det_data.get_value(index_value=max_percentile_value, data_name=ALL_DATA_KEYWORD)
    min_all_value_mc = mc_data.get_value(index_value=min_percentile_value, data_name=ALL_DATA_KEYWORD)
    max_all_value_mc = mc_data.get_value(index_value=max_percentile_value, data_name=ALL_DATA_KEYWORD)

    """------------------------Run FRCE analysis------------------------------------------------------------------"""
    frce = FrequencyRestorationControlError()
    frce.set_aceol_data(data=imbalance_volume_data_in_time_range[region_to_investigate])
    # frce.set_percentiles()
    frce.set_adjusted_percentiles()
    dt_1_min, dt_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_det,
                                                                 negative_fr=min_all_value_det)
    dt_2_min, dt_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_det,
                                                                 negative_fr=min_all_value_det)
    mc_1_min, mc_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_mc,
                                                                 negative_fr=min_all_value_mc)
    mc_2_min, mc_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=max_all_value_mc,
                                                                 negative_fr=min_all_value_mc)
    sf_1_min, sf_1_max = frce.level_1.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=frce.level_1.percentile.upper_value,
                                                                 negative_fr=frce.level_1.percentile.lower_value)
    sf_2_min, sf_2_max = frce.level_2.get_frr_percent_over_level(data=frce.aceol_data,
                                                                 positive_fr=frce.level_2.percentile.upper_value,
                                                                 negative_fr=frce.level_2.percentile.lower_value)
    det_label = f"{det_data.description}, {ALL_DATA_KEYWORD}"
    mc_label = f"{mc_data.description}, {ALL_DATA_KEYWORD}"
    level_1_dict = {INITIAL_ALLOWED: f"{frce.level_1.percent_of_time_from_year}%",
                    INITIAL_UNCORRECTED: f"{frce.level_1.percentage_over_aceol_data:.2f}%",
                    det_label: f"{(dt_1_min + dt_1_max):.4f}%",
                    mc_label: f"{(mc_1_min + mc_1_max):.4f}%"}
    level_2_dict = {INITIAL_ALLOWED: f"{frce.level_2.percent_of_time_from_year}%",
                    INITIAL_UNCORRECTED: f"{frce.level_2.percentage_over_aceol_data:.2f}%",
                    det_label: f"{(dt_2_min + dt_2_max):.4f}%",
                    mc_label: f"{(mc_2_min + mc_2_max):.4f}%"}
    frce_report = pd.DataFrame([level_1_dict, level_2_dict], index=['Level 1', 'Level 2'])

    level_1_self_values = {INITIAL_ALLOWED: f"{frce.level_1.percent_of_time_from_year}%",
                           TARGET: f"{frce.level_1.target_value}MW",
                           LOWER_BOUND: f"{frce.level_1.percentile.lower_value:.2f}MW",
                           UPPER_BOUND: f"{frce.level_1.percentile.upper_value:.2f}MW",
                           EXCESS_WHEN_APPLIED: f"{(sf_1_min + sf_1_max):.4f}%"}
    level_2_self_values = {INITIAL_ALLOWED: f"{frce.level_2.percent_of_time_from_year}%",
                           TARGET: f"{frce.level_2.target_value}MW",
                           LOWER_BOUND: f"{frce.level_2.percentile.lower_value:.2f}MW",
                           UPPER_BOUND: f"{frce.level_2.percentile.upper_value:.2f}MW",
                           EXCESS_WHEN_APPLIED: f"{(sf_2_min + sf_2_max):.4f}%"}
    frce_self_report = pd.DataFrame([level_1_self_values, level_2_self_values], index=['Level 1', 'Level 2'])

    dt_1_passed = frce.level_1.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_det,
                                                            negative_fr=min_all_value_det)
    dt_2_passed = frce.level_2.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_det,
                                                            negative_fr=min_all_value_det)
    mc_1_passed = frce.level_1.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_mc,
                                                            negative_fr=min_all_value_mc)
    mc_2_passed = frce.level_2.check_frr_percent_over_level(data=frce.aceol_data,
                                                            positive_fr=max_all_value_mc,
                                                            negative_fr=min_all_value_mc)
    det_magic_string = "none of the levels"
    mc_magic_string = det_magic_string
    if dt_1_passed and dt_2_passed:
        det_magic_string = "both of the levels"
    elif dt_1_passed:
        det_magic_string = frce.level_1.level_name
    elif dt_2_passed:
        det_magic_string = frce.level_2.level_name
    if mc_1_passed and mc_2_passed:
        mc_magic_string = "both of the levels"
    elif mc_1_passed:
        mc_magic_string = frce.level_1.level_name
    elif mc_2_passed:
        mc_magic_string = frce.level_2.level_name
    """------------------------Generate the report----------------------------------------------------------------"""
    print("Results: reserve capacities")
    print(overall_report.to_markdown())
    print("Comparison with frce results")
    print(frce_report.to_markdown())
    # reset the index if contemporary results were loaded from the file
    if 'Unnamed: 0' in list(overall_report.columns):
        overall_report.set_index(['Unnamed: 0'], inplace=True)
    heading = 'Recommendations of minimum reserve capacity at SOR level'
    methodology_calc_heading = "Calculation of reserve capacities"
    region_name_value = region_to_investigate
    if region_name_value.lower() == 'baltics':
        region_name_value = 'Baltic'
    max_percentile_value = max(ANALYSIS_PERCENTILES.values())
    methodology_calc = (f"Calculation is based on the at least 1 "
                        f"year data ending not later than 6 months "
                        f"before the time when current report was created. The results (positive (+) and "
                        f"negative values (-)) are "
                        f"summed over the SOR (here noted as {region_name_value} region). "
                        f"Calculation is performed in two ways:\n"
                        f"a) Deterministic approach (Det.) finds {max_percentile_value * 100}% percentiles  from ACEol"
                        f" (Area Control Error Open loop) data directly, assuming that it is normally distributed.\n"
                        f"b) Probabilistic approach (MC) samples the input data {NUMBER_OF_SAMPLES} times per "
                        f"simulation ({NUMBER_OF_SIMULATIONS} in total). From the results mean of  "
                        f"{max_percentile_value * 100}% "
                        f"percentiles is taken with 1.96 * standard deviation indicating confidence level of 95% "
                        f"({PLUS_MINUS}) over all the {NUMBER_OF_SIMULATIONS} simulations.")
    methodology_frce_heading = "Comparison of results with FRCE values"
    methodology_frce = (f"Here FRCE (Frequency Restoration Control Error) represents the absolute (in positive and "
                        f"negative direction) limit values for the ACE:\n"
                        f"\n"
                        f"                                                  "
                        f"ACE = ACEol + (mFRR + aFRR)"
                        f"                                                  "
                        f"[1]\n\n"
                        f"Here the following cases are considered:\n"
                        f"a) For {frce.level_1.level_name} the residual value (ACE) cannot exceed "
                        f"{frce.level_1.target_value}{frce.level_1.unit} for more than "
                        f"{frce.level_1.percent_of_time_from_year}% of time intervals of the year\n"
                        f"b) For {frce.level_2.level_name} the residual value (ACE) cannot exceed "
                        f"{frce.level_2.target_value}{frce.level_2.unit} for more than "
                        f"{frce.level_2.percent_of_time_from_year}% of time intervals of the year\n"
                        f"For estimating the amount of exceed cases, the found reserve capacities (varying from 0 to "
                        f"maximum value) were applied to the uncorrected (ACEol) data. From the results extreme "
                        f"cases, that exceed the targets set by levels (a and b) were extracted. Their amount is "
                        f"represented as percentage of the time intervals of the year in corresponding table.\n"
                        f"In order to illustrate the reserve capacities needed to fulfill the FRCE requirements, the "
                        f"reverse process by leveling the data with FRCE target values and determining the needed "
                        f"minimal percentiles numerically were carried out. Corresponding results are depicted in table"
                        f" 3 by the FRCE levels. These values, however, should be regarded informative only as "
                        f"they are based solely on the data and do not represent the real situation.")
    list_of_methodologies = {methodology_calc_heading: methodology_calc, methodology_frce_heading: methodology_frce}
    main_summary_heading = "main_summary"
    main_summary = (f"Based on the instructions, the minimum reserve capacity is found from the summed results"
                    f" (positive and negative values at SOR level (presented on the left side figures below). "
                    f"Analysis solely based on negative and positive values is here only for the reference. "
                    f"(middle and right side figure)\n"
                    f"Method (a) gave for the positive capacity {max_all_value_det:.1f}MW and method "
                    f"(b) {max_all_value_mc:.1f}MW with {max(max_all_value_det, max_all_value_mc):.1f}MW as general "
                    f"recommendation. "
                    f"For the negative values, method (a) produced {min_all_value_det:.1f}MW and method "
                    f"(b) {min_all_value_mc:.1f}MW with {min(min_all_value_det, min_all_value_mc):.1f}MW as "
                    f"general recommendation.\n")
    frce_summary_heading = "frce_summary"
    frce_summary = (f"From the results it is possible to conclude that values proposed by Method (a) satisfy "
                    f"{det_magic_string} and values proposed by Method (b) satisfy {mc_magic_string}")
    list_of_summaries = {main_summary_heading: main_summary, frce_summary_heading: frce_summary}
    deterministic_fig_title = (f'Deterministic solution, blue lines represent values at {max_percentile_value * 100}%.'
                               f'Figures from left: a) All data summed, b) extracted positive values, c) extracted '
                               f'negative values.Amount of extreme cases is indicated on the top of the figure')
    mc_fig_title = (f"Monte Carlo results, blue lines represent values at {max_percentile_value * 100}%, dashed lines "
                    f"represent {PLUS_MINUS}95%. Figures from left: a) All data summed, b) extracted positive values, "
                    f"c) extracted negative values. Amount of extreme cases is indicated on the top of the figure")
    input_data_fig_title = (f"Input imbalance volume data for {region_name_value} "
                            f"from {start_time.strftime(DATE_FORMAT_FOR_REPORT)} to "
                            f"{end_time.strftime(DATE_FORMAT_FOR_REPORT)}. Data taken from Baltic Transparency Dashboard")
    table_title = (f"Capacities with different methods (MC: Monte Carlo, Det. deterministic) "
                   f"and different datasets (all values in MW)")
    frce_table_title = (f'Comparison of FRCE levels (percentage value in column "{INITIAL_ALLOWED}"  and percentage of '
                        f'time moments when ACEol exceeded the target in column "{INITIAL_UNCORRECTED}". After '
                        f'applying the values found during the calculation of the reserve capacities, the percentage '
                        f'of the time moments when residual value exceeded the target of the level is shown in '
                        f'columns "{det_label}" and "{mc_label}" respectively.')
    frce_self_table_title = (f'Minimum values for reserve capacities considering the requirements of FRCE levels. '
                             f'Column "{INITIAL_ALLOWED}" maximum allowed time moments when ACE can exceed '
                             f'the value in column "{TARGET}". Numerically minimum values are in columns '
                             f'("{UPPER_BOUND}") and ("{LOWER_BOUND}"). '
                             f'Column "{EXCESS_WHEN_APPLIED}" shows percentage of time moments '
                             f'when ACE exceeds the value in "{TARGET}" after applying up and down values.')
    reference_list = {'[1]': "ENTSO-E Proposal for the Regional Coordination Centres' task 'regional sizing of "
                             "reserve capacity' in accordance with Article 37(1)(j) of the regulation (EU) 2019/943 "
                             "of the European Parliament and of the Council of 5 June 2019 on the internal market for "
                             "electricity",
                      '[2]': "Baltic Load-Frequency Control block concept document, 31.12.2020"}
    list_of_figures = {}
    if draw_raw_data_image:
        list_of_figures[input_data_fig_title] = INPUT_DATA_FIGURE_NAME
    list_of_figures[deterministic_fig_title] = DETERMINISTIC_FIGURE_NAME
    list_of_figures[mc_fig_title] = MC_FIGURE_NAME
    list_of_tables = {table_title: overall_report,
                      frce_table_title: frce_report,
                      frce_self_table_title: frce_self_report}
    try:
        report_date = calculation_date.strftime("%d-%m-%Y")
    except ValueError:
        report_date = pd.Timestamp("today").strftime("%d-%m-%Y")
    report_name = f"report_of_regional_sizing_at_SOR_level_for_Baltics_from_{report_date}"
    if PY_SIZING_OUTPUT_TO_WORD:
        report_word = generate_report_word(heading_string=heading,
                                           start_date=start_time,
                                           end_date=end_time,
                                           methodologies=list_of_methodologies,
                                           region_list=[region_to_investigate],
                                           summaries=list_of_summaries,
                                           references=reference_list,
                                           images=list_of_figures,
                                           tables=list_of_tables,
                                           # file_name=r"E:\margus.ratsep\sizing_of_reserves\reports\test_report.docx",
                                           date_today=calculation_date,
                                           )
        if report_word:
            target_stream = BytesIO()
            report_word.save(target_stream)
            # report_word = BytesIO(report_word)
            target_stream.name = report_name.removesuffix(".docx") + '.docx'
            save_file_to_minio_with_link(target_stream)

    if PY_SIZING_OUTPUT_TO_PDF:

        report_pdf = generate_report_pdf(heading_string=heading,
                                         start_date=start_time,
                                         end_date=end_time,
                                         methodologies=list_of_methodologies,
                                         region_list=[region_to_investigate],
                                         summaries=list_of_summaries,
                                         references=reference_list,
                                         images=list_of_figures,
                                         tables=list_of_tables,
                                         # file_name=report_name
                                         date_today=calculation_date,
                                         )
        if report_pdf:
            if report_pdf:
                report_pdf = BytesIO(report_pdf)
                report_pdf.name = report_name.removesuffix(".pdf") + '.pdf'
                save_file_to_minio_with_link(report_pdf)

    logger.info(f"Cleaning up")
    delete_file_from_local_storage(INPUT_DATA_FIGURE_NAME)
    delete_file_from_local_storage(DETERMINISTIC_FIGURE_NAME)
    delete_file_from_local_storage(MC_FIGURE_NAME)
    logger.info("Done")
