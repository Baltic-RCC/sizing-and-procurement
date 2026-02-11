import logging
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from py.sizing_of_reserves.report_generation import PLUS_MINUS

LEFT_MARGIN_COLOR = 'r'
RIGHT_MARGIN_COLOR = 'g'
CENTRE_LINE_COLOR = 'b'
CENTRE_LINE_STYLE = '-'
MARGIN_LINE_STYLE = ':'
DEFAULT_X_LABEL = 'Imbalance (MW)'

FRCE_LEVEL_1_POWER = 35
FRCE_LEVEL_1_PERCENTAGE = 30
FRCE_LEVEL_1_UNIT = 'MW'
FRCE_LEVEL_2_POWER = 65
FRCE_LEVEL_2_PERCENTAGE = 5
FRCE_LEVEL_2_UNIT = 'MW'

logger = logging.getLogger(__name__)

class CapacityDataSeriesCenterLine:
    """
    Auxiliary class for storing vertical line information
    """

    def __init__(self, percentile: int = 0, deviation: float = 0, center_line: float | None = None, label: str = None):
        """
        Constructor
        :param center_line: vertical line location in x-axis
        :param percentile: identification of center line
        :param deviation: if center line has a confidence level then this represents the +/- values
        :param label: name of the center line in plots
        """
        self.center_line = center_line
        self.deviation = deviation
        self.margins = {}
        if self.deviation > 0:
            self.margins = {'-': center_line - deviation, '+': center_line + deviation}
        self.percentile = percentile
        self.label = label


class CapacityDataSeries:
    """
    Auxiliary class for storing single data series information for plotting
    """

    def __init__(self, input_data: pd.DataFrame, column: str, label: str, title: str, center_lines=None):
        """

        :param input_data: dataframe containing the data for plotting
        :param column: name of the column for plotting
        :param label: name of the data series from the input data that will be plotted
        :param title: figure title
        :param center_lines: dictionary of vertical lines for the figure
        """
        if center_lines is None:
            center_lines = {}
        self.data = input_data
        self.label = label
        self.column = column
        self.title = title
        self.center_lines = center_lines

    def get_x_lim_values(self):
        """
        Returns min and max values along the x-axis for the data set that will be plotted
        :return: tuple of min and max value "along the x-axis"
        """
        return min(self.data[self.column]), max(self.data[self.column])


class MinimumCapacityData:
    """
    Auxiliary class for storing data for plotting the results
    """

    def __init__(self, input_data=None, description: str = None):
        """
        Constructor

        :param input_data: dictionary of capacity data series as {series name: capacity series}
        :param description: name of the data (used later as part of the title in report dataframe)
        """
        if input_data is None:
            input_data = []
        self.data = input_data
        self.description = description

    def get_number_of_graphs(self):
        """
        Returns number of data series stored within
        :return: number of data series
        """
        return len(self.data)

    def get_x_limit_values(self):
        """
        Finds overall x limits (all graphs share the same scale)

        :return: tuple of min max values
        """
        x_min = 0
        x_max = 0
        for data_series in self.data:
            new_x_min, new_x_max = data_series.get_x_lim_values()
            x_min = min(x_min, new_x_min)
            x_max = max(x_max, new_x_max)
        return x_min, x_max

    def plot_data(self, file_name: str):
        """
        Plots all the data series in one graph, one subplot per data series
        All data series are plotted as density functions with vertical lines coming from center lines

        :param file_name: name of the file where to store the result image
        :return: None
        """
        margin_colors = [LEFT_MARGIN_COLOR, RIGHT_MARGIN_COLOR]
        x_min, x_max = self.get_x_limit_values()
        _, axes = plt.subplots(1, self.get_number_of_graphs(), squeeze=False)
        image_counter = 0
        for data_series in self.data:
            data_density = data_series.data[data_series.column].plot.kde(ax=axes[0][image_counter],
                                                                         label=data_series.label)
            for center_line in data_series.center_lines:
                data_density.axvline(x=center_line.center_line,
                                     color=CENTRE_LINE_COLOR,
                                     linestyle=CENTRE_LINE_STYLE,
                                     label=center_line.label)
                margin_counter = 0
                for margin in center_line.margins:
                    margin_color_id = margin_counter % len(margin_colors)
                    data_density.axvline(x=center_line.margins[margin],
                                         color=margin_colors[margin_color_id],
                                         linestyle=MARGIN_LINE_STYLE,
                                         label=margin)
                    margin_counter += 1
            data_density.set_title(data_series.title)
            data_density.set_xlim(x_min, x_max)
            data_density.grid()
            # data_density.legend()
            data_density.set_xlabel(DEFAULT_X_LABEL)
            image_counter += 1
        # plt.show()
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches='tight')

    def get_report(self, index_list: {}):
        """
        Generates a report in a form where description + data series name is column and center line values are
        categorized by the sequence numbers of the data series. In case of absence, value is presented as -

        :param index_list: dictionary of indexes and percentiles to be used
        :return: dataframe with results
        """
        result_columns = {}
        for data_series in self.data:
            data_label = f"{self.description}, {data_series.label}"
            data_values = ['-'] * len(index_list)
            for i, index in enumerate(index_list.keys()):
                for center_line in data_series.center_lines:
                    if center_line.percentile == index_list[index]:
                        if center_line.deviation > 0:
                            data_values[i] = f"{center_line.center_line:.1f}{PLUS_MINUS}{center_line.deviation:.1f}"
                        else:
                            data_values[i] = f"{center_line.center_line:.1f}"
            result_columns[data_label] = data_values
        return pd.DataFrame(result_columns, index=index_list)

    def get_value(self, index_value, data_name):
        """
        Gets a single center line value

        :param index_value: sequence number of center line
        :param data_name: data series name
        :return: value as float
        """
        for data_series in self.data:
            if data_series.label == data_name:
                for center_line in data_series.center_lines:
                    if center_line.percentile == index_value:
                        return center_line.center_line
        return 0


class FrequencyRestorationControlErrorPercentile:
    """
    Data class to store data about the percentiles
    """

    def __init__(self, min_q: float, max_q: float, span: float):
        """
        Initialization

        :param min_q: lower (negative) value
        :param max_q: upper (positive) value
        :param span: percentage of time between min_q and max_q
        """
        self.lower_value = min_q
        self.upper_value = max_q
        self.time_span = span


class FrequencyRestorationControlErrorLevel:
    """
    Depicts the FRCE limit value consisting of limit value, its unit and the percentage of time
    intervals of the year when residual value (ACE=ACEol - FRR) is allowed to surpass the value
    """

    def __init__(self, level: str, error_value: float, error_percentage: float, error_unit: str = 'MW'):
        """
        Constructor

        :param level: Level name, usually 'Level 1' etc.
        :param error_value: target value
        :param error_percentage: percentage of the time instances in year when ACE can exceed the error value.
        :param error_unit: error value unit (usually MW)
        """
        self.level_name = level
        self.target_value = error_value
        self.percent_of_time_from_year = error_percentage
        self.unit = error_unit
        self.percentage_over_aceol_data = 0
        self.percentile = None

    @property
    def portion_of_percentage(self):
        """
        Return percentage of the allowed time intervals from the year as portion from 0 to 1

        :return: decimal indicating the portion
        """
        return self.percent_of_time_from_year / 100

    def set_percentage_over_aceol_data(self, data: pd.Series):
        """
        Calculate the percentage of the values that are exceeding the target value

        :param data: aceol data
        :return: None
        """
        self.percentage_over_aceol_data = 100 * len(data[abs(data) > self.target_value]) / len(data)

    def calculate_symmetric_percentiles(self, percentile_value, data: pd.Series):
        """
        Finds +, - percentiles based on the input (half from the left end, half from the right end)

        :param percentile_value: input percentile value
        :param data: input data
        :return: dictionary containing percentile, its found values and coverage by target value
        """
        negative_position = percentile_value / 2
        positive_position = 1 - negative_position
        positive_percentile = data.quantile(q=positive_position)
        negative_percentile = data.quantile(q=negative_position)
        off_threshold = 100 * len(data[(data > positive_percentile + self.target_value) |
                                       (data < negative_percentile - self.target_value)]) / len(data)
        in_threshold = 100 * len(data[(data <= positive_percentile + self.target_value) &
                                      (data >= negative_percentile - self.target_value)]) / len(data)
        return {'percent': percentile_value,
                'lower': negative_percentile,
                'upper': positive_percentile,
                'off': off_threshold,
                'in': in_threshold}

    def calculate_all_percentiles(self, data: pd.Series):
        """
        Finds all percentiles in a range

        :param data: input data
        :return: dataframe containing percentile, its found values and coverage by target value
        """
        data = data.dropna()
        max_portion = max(0.49, self.portion_of_percentage)
        percentile_values = np.arange(0, max_portion, 0.001)
        outputs = []
        for percentile_value in percentile_values:
            values = self.calculate_symmetric_percentiles(percentile_value, data)
            outputs.append(values)
        return pd.DataFrame(outputs)

    def set_adjusted_percentiles_to_aceol_data(self, data: pd.Series):
        """
        Finds closest match (exponential function) to cover the cases
        x + self.target_value < self.percent_of_time_from_year

        :param data: input data
        :return: None
        """
        data = data.dropna()
        full_length = len(data)
        brute_force_approach = self.calculate_all_percentiles(data)
        closest_match = brute_force_approach.iloc[(brute_force_approach['off'] - self.percent_of_time_from_year)
                                                  .abs()
                                                  .argsort()[:1]]
        negative_quantile = float(closest_match['lower'].iloc[0])
        positive_quantile = float(closest_match['upper'].iloc[0])
        covered = float(closest_match['in'].iloc[0])
        not_covered = float(closest_match['off'].iloc[0])
        percentage_of_time = (100 * len(data[(data >= negative_quantile) & (data <= positive_quantile)]) / full_length)
        self.percentile = FrequencyRestorationControlErrorPercentile(min_q=negative_quantile,
                                                                     max_q=positive_quantile,
                                                                     span=percentage_of_time)
        logger.info(f"Closest match: +:{positive_quantile:.2f}, -:{negative_quantile:.2f}, "
                    f"Over: {not_covered:.2f}%, within: "
                    f"{covered:.2f}%")

    def get_frr_percent_over_level(self, data: pd.Series, positive_fr: float = 0, negative_fr: float = 0):
        """
        Gets the percentage of the values exceeding the given target level after applying the frequency restoration
        values (positive and negative) to the input (aceol) data. Note that the assumption is that data is centered
        around zero

        :param data: imbalance_volume_data as Series
        :param positive_fr: positive frequency restoration
        :param negative_fr: negative frequency restoration
        :return: percentages: in negative direction, in positive direction
        """
        # Assume that the normal distribution is centered around 0
        data = data.dropna()
        data_length = len(data)
        # TODO come up with more general approach
        if negative_fr <= 0 <= positive_fr:
            positive_data = data[data > positive_fr] - positive_fr
            negative_data = data[data < negative_fr] - negative_fr
            positive_percentage = 100 * len(positive_data[positive_data > abs(self.target_value)]) / data_length
            negative_percentage = 100 * len(negative_data[negative_data < (-1) * abs(self.target_value)]) / data_length
            return negative_percentage, positive_percentage
        return None, None

    def check_frr_percent_over_level(self, data: pd.Series, positive_fr: float = 0, negative_fr: float = 0):
        """
        Checks if total percentage of exceeding values is smaller than allowed percentage
        (self.percent_of_time_from_year)

        :param data: aceol data series
        :param positive_fr: frequency restoration in positive direction
        :param negative_fr: frequency restoration in negative direction
        :return: True if the summed output was smaller, false otherwise
        """
        negative_percent, positive_percent = self.get_frr_percent_over_level(data,
                                                                             positive_fr=positive_fr,
                                                                             negative_fr=negative_fr)
        if negative_percent is not None and positive_percent is not None:
            return negative_percent + positive_percent <= self.percent_of_time_from_year
        return False


class FrequencyRestorationControlError:
    """
    Dataclass for storing FRCE related data. Currently, contains two fixed levels as level 1 and level 2
    """

    def __init__(self,
                 level_1_value: float = FRCE_LEVEL_1_POWER,
                 level_2_value: float = FRCE_LEVEL_2_POWER,
                 level_1_percentage: float = FRCE_LEVEL_1_PERCENTAGE,
                 level_2_percentage: float = FRCE_LEVEL_2_PERCENTAGE,
                 level_1_unit: str = FRCE_LEVEL_1_UNIT,
                 level_2_unit: str = FRCE_LEVEL_2_UNIT):
        """
        Init method

        :param level_1_value: allowed  ABS(ACE = ACEol - FRR)  (level_1_percentage)% of a year
        :param level_2_value:  allowed  ABS(ACE = ACEol - FRR)  (level_2_percentage)% of a year
        :param level_1_percentage: percentage of time instances of the year that where residual value
                                   (level_1_value: after applying FRR to ACEol) is allowed
        :param level_2_percentage: percentage of time instances of the year that where residual value
                                   (level_2_value: after applying FRR to ACEol) is allowed
        :param level_1_unit: unit for level_1_value
        :param level_2_unit: unit for level_2_value
        """
        self.level_1 = FrequencyRestorationControlErrorLevel(level='Level 1',
                                                             error_value=level_1_value,
                                                             error_percentage=level_1_percentage,
                                                             error_unit=level_1_unit)
        self.level_2 = FrequencyRestorationControlErrorLevel(level='Level 2',
                                                             error_value=level_2_value,
                                                             error_percentage=level_2_percentage,
                                                             error_unit=level_2_unit)
        # Analysis results
        self.aceol_data = None

    def set_aceol_data(self, data: pd.Series):
        """
        Sets aceol data and calculates the initial percentage of the values over level 1 and level 2

        :param data: aceol data (raw form)
        :return: None
        """
        self.aceol_data = data
        self.level_1.set_percentage_over_aceol_data(data)
        self.level_2.set_percentage_over_aceol_data(data)

    def set_adjusted_percentiles(self):
        """
        Sets the percentiles (level_1_percentile and level_2_percentile) to the imbalance_volume_data
        (self.imbalance_volume_data)

        :return: None
        """
        if self.aceol_data is not None:
            self.level_1.set_adjusted_percentiles_to_aceol_data(self.aceol_data)
            self.level_2.set_adjusted_percentiles_to_aceol_data(self.aceol_data)


def ceil(input_value: float, decimals: int = 0):
    """
    Extends built-in ceil to ceil by number of decimals indicated by decimals:
    floor(10.5612, 2) = 10.57

    :param input_value: value to be rounded up
    :param decimals: number of digits after separator
    :return: rounded value
    """
    decimals_value = pow(10, decimals)
    return math.ceil(input_value * decimals_value) / decimals_value


def floor(input_value: float, decimals: int = 0):
    """
    Extends built-in floor to floor by number of decimals indicated by decimals:
    floor(10.5678, 2) = 10.56

    :param input_value: value to be rounded down
    :param decimals: number of digits after separator
    :return: rounded value
    """
    decimals_value = pow(10, decimals)
    return math.floor(input_value * decimals_value) / decimals_value
