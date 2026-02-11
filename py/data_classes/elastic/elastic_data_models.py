import itertools
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import isodate

from py.common.functions import check_dict_to_dataclass
from py.common.time_functions import convert_string_to_datetime, convert_datetime_to_string_utc, time_delta_to_str, \
    check_and_parse_duration
from py.common.ref_constants import SERIES_VALID_FROM_KEY, SERIES_VALID_TO_KEY, DOMAIN_MRID_KEY, TYPE_NAME_KEY, \
    OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY, DIRECTION_NAME_KEY, PERCENTAGE_VALUE_KEY, SERIES_VALID_FROM_F_KEY, \
    SERIES_VALID_TO_F_KEY, POINT_RESOLUTION_KEY, CALC_DATE_KEY, DATA_PERIOD_END_KEY, DATA_PERIOD_START_KEY, PROCESS_TYPE_KEY, \
    MESSAGE_TYPE_KEY, CURVE_TYPE_KEY, BUSINESS_TYPE_KEY, PRODUCT_KEY
from py.data_classes.task_classes import QuantileArray, Domain, TIME_KEYS, QUANTILE_KEYS, MessageCommunicator
from py.data_classes.enums import MeasurementUnitType, EnergyProductType, \
    CurveType, ProcessType, MessageType, BusinessType, ProcurementCalculationType

logger = logging.getLogger(__name__)

SERIES_TIME_KEYS = [SERIES_VALID_FROM_KEY, SERIES_VALID_TO_KEY]
ADDITIONAL_TIME_ATTRIBUTES = [DATA_PERIOD_START_KEY, DATA_PERIOD_END_KEY, CALC_DATE_KEY]
TIME_ATTRIBUTES = list(itertools.chain(ADDITIONAL_TIME_ATTRIBUTES, TIME_KEYS))

ROUND_VALUES = 1


TIME_SLICE_KEYS = [SERIES_VALID_FROM_F_KEY, SERIES_VALID_TO_F_KEY, POINT_RESOLUTION_KEY]

BID_TYPE_KEYS = [DOMAIN_MRID_KEY, TYPE_NAME_KEY, DIRECTION_NAME_KEY]
ATC_TYPE_KEYS = [OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY]


BID_TIMESERIES_KEYS = list(itertools.chain(BID_TYPE_KEYS, TIME_SLICE_KEYS,
                                           [PERCENTAGE_VALUE_KEY], QUANTILE_KEYS))
ATC_TIMESERIES_KEYS = list(itertools.chain(ATC_TYPE_KEYS, TIME_SLICE_KEYS,
                                           [PERCENTAGE_VALUE_KEY], QUANTILE_KEYS))
BID_PRODUCT_KEYS = list(itertools.chain(BID_TYPE_KEYS, TIME_KEYS, QUANTILE_KEYS))
ATC_PRODUCT_KEYS = list(itertools.chain(ATC_TYPE_KEYS, TIME_KEYS, QUANTILE_KEYS))

@dataclass
class DataToXMLPoint:
    primary_quantity: float
    position: int = 1
    secondary_quantity: float = None

    def round_values(self, rounding_decimals):
        if rounding_decimals:
            self.primary_quantity = round(self.primary_quantity, rounding_decimals)
            if self.secondary_quantity:
                self.secondary_quantity = round(self.secondary_quantity, rounding_decimals)

    def __post_init__(self):
        self.round_values(rounding_decimals=ROUND_VALUES)


@dataclass
class DataPointType:
    name: str                                       # Name of the datapoint
    description: str = None                         # Free form description
    code: str = None                                # EIC code if applicable


@dataclass
class DataPoint:
    value: float | str                              # Store actual value
    measurement_unit: str | MeasurementUnitType     # Store measurement type
    position: int = 1                               # Store relative position

    def __post_init__(self):
        # self.measurement_unit = set_enum(input_value=self.measurement_unit,type_enum=MeasurementUnitType)
        pass


@dataclass
class QuantileSpacing:

    start_percentage: float
    step_percentage: float
    stop_percentage: float

    @staticmethod
    def from_spacing(spacing: QuantileArray):
        return QuantileSpacing(start_percentage=spacing.spacing_start_value,
                               step_percentage=spacing.spacing_step_size,
                               stop_percentage=spacing.spacing_end_value)


def generate_mrid():
    """
    Generates mrid
    Maybe generate something from the fields
    :return:
    """
    return str(uuid.uuid4())


@dataclass
class CalculationPoint:
    # common
    calculation_date:  str | datetime                   # When result was calculated
    version_number: float | int | str                   # Version number
    available: DataPoint                                # Primary value of result (usually in MW)
    percentage_level: DataPoint                         # Secondary value of result, usually percentage
    LFC_block: Domain                                   # Area where the value is effective
    valid_from: str | datetime = None                   # Start time form where result is valid
    valid_to: str | datetime = None                     # End time to when the result is valid
    _series_valid_from: str | datetime = None
    _series_valid_to: str | datetime = None
    point_resolution: str = None                        # The resolution for the point
    number_of_points: int | str = None                      # Number of points
    mrid: str = None
    result_type: str = None                             # Either ATC or NCPB
    data_period_start: str | datetime = None            # Start point from where the data was used
    data_period_end: str | datetime = None              # End point to where the data was used
    valid_resolution: str = None
    data_resolution: str = 'P15M'                       # Resolution of the data used for calculation
    type: DataPointType = None                          # Specify which kind of product it is
    quantile_spacing: QuantileSpacing = None            # Specify X and Y parameters used for calculating
    process_type: str | ProcessType = None              # EIC code for process.processType
    message_type: str | MessageType = None              # EIC code for type
    business_type: str | BusinessType = None            # EIC code for businessType
    curve_type: str | CurveType = None                  # EIC code for curve type
    product: str | EnergyProductType = None             # EIC code for product
    sender: Domain | MessageCommunicator = None         # Sender of the result
    receiver: Domain | MessageCommunicator = None       # Receiver of the result, needed for exceeded
    description: str = None

    def _set_series_valid_to(self):
        if self._series_valid_to is None and self.number_of_points:
            if math.isnan(self.number_of_points):
                self.number_of_points = 1
            no_points = int(self.number_of_points)
            self._series_valid_to = self.series_valid_from + no_points * self.get_point_resolution()
        return self._series_valid_to

    @property
    def series_valid_from(self):
        if self._series_valid_from is None:
            self.convert_string_to_time(TIME_KEYS)
            minus_value = max(self.available.position - 1, 0)
            p_resolution = check_and_parse_duration(self.get_point_resolution())
            valid_from = convert_string_to_datetime(self.valid_from)
            self._series_valid_from = valid_from - (minus_value * p_resolution)
        return self._series_valid_from

    @property
    def series_valid_to(self):
        return self._set_series_valid_to()

    @series_valid_to.setter
    def series_valid_to(self, new_value):
        self._series_valid_to = new_value

    @series_valid_from.setter
    def series_valid_from(self, new_value):
        self._series_valid_from = new_value

    def _series_period_to_point_period(self):
        self.convert_time_to_string(attributes=[*SERIES_TIME_KEYS, *TIME_KEYS])
        if self.valid_from is not None:
            self._set_series_valid_to()
        if self.valid_from is None and self._series_valid_from:
            self.valid_from = self._series_valid_from + (self.available.position - 1) * self.get_point_resolution()
        if self.valid_to is None and self.valid_from is not None:
            self.valid_to = self.valid_from + self.get_point_resolution()

    def sanitize_enum_inputs(self, enum_dict: dict = None):
        if enum_dict is None:
            enum_dict = {PROCESS_TYPE_KEY: ProcessType,
                         MESSAGE_TYPE_KEY: MessageType,
                         BUSINESS_TYPE_KEY: BusinessType,
                         CURVE_TYPE_KEY:CurveType,
                         PRODUCT_KEY: EnergyProductType}
        for field_name, field_type in enum_dict.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                if current_value is None or (not isinstance(current_value, str) and math.isnan(current_value)):
                    continue
                try:
                    current_value = field_type.value_of(str(current_value))
                except ValueError:
                    pass
                if isinstance(current_value, Enum):
                    setattr(self, field_name, str(current_value.value))
                else:
                    setattr(self, field_name, str(current_value))

    def __post_init__(self):
        self.convert_time_to_string()
        self.LFC_block = check_dict_to_dataclass(Domain, self.LFC_block)
        self.available = check_dict_to_dataclass(DataPoint, self.available)
        self.percentage_level = check_dict_to_dataclass(DataPoint, self.percentage_level)
        self.type = check_dict_to_dataclass(DataPointType, self.type)
        self.quantile_spacing = check_dict_to_dataclass(QuantileSpacing, self.quantile_spacing)
        self.sanitize_enum_inputs()
        self.get_valid_resolution()
        self.mrid = self.mrid or generate_mrid()
        self.version_number = str(self.version_number) if self.version_number is not None else self.version_number
        if isinstance(self.sender, MessageCommunicator):
            self.sender = self.sender.get_domain()
        self.sender = check_dict_to_dataclass(Domain, self.sender)
        self._series_period_to_point_period()
        self.percentage_level.position = self.available.position
        self.get_number_of_points()

    def convert_time_to_string(self, attributes: list = None):
        if not attributes:
            attributes = TIME_ATTRIBUTES
        for attribute in attributes:
            if hasattr(self, attribute):
                setattr(self, attribute, convert_datetime_to_string_utc(getattr(self, attribute)))

    def convert_string_to_time(self, attributes: list = None):
        if not attributes:
            attributes = TIME_ATTRIBUTES
        for attribute in attributes:
            if hasattr(self, attribute):
                setattr(self, attribute, convert_string_to_datetime(getattr(self, attribute)))

    def get_point_resolution(self):
        self.point_resolution = check_and_parse_duration(self.point_resolution)
        if (self.point_resolution is None or
                (isinstance(self.point_resolution, float) and math.isnan(self.point_resolution))):
            self.convert_string_to_time(TIME_KEYS)
            self.point_resolution = time_delta_to_str(self.valid_to - self.valid_from)
        return self.point_resolution

    def get_valid_resolution(self):
        self.valid_resolution = check_and_parse_duration(self.valid_resolution)
        if self.valid_resolution is None:
            self.valid_resolution = time_delta_to_str(convert_string_to_datetime(self.series_valid_to) -
                                                      convert_string_to_datetime(self.series_valid_from))
        return self.valid_resolution

    def get_number_of_points(self):
        if self.number_of_points is None:
            if self._series_valid_from and self._series_valid_to:
                if res := self.get_point_resolution():
                    self.convert_string_to_time(SERIES_TIME_KEYS)
                    self.number_of_points = int((self._series_valid_to - self.series_valid_from) / res)
            else:
                # for backward compatibility initiate this as 1 (P1D
                self.number_of_points = 1

    def generate_spacing_id(self):
        full_id = ''
        if self.quantile_spacing is not None and isinstance(self.quantile_spacing, QuantileSpacing):
            id_fields = [self.quantile_spacing.start_percentage,
                         self.quantile_spacing.step_percentage,
                         self.quantile_spacing.stop_percentage]
            id_fields = [str(id_field) for id_field in id_fields if id_field]
            full_id = '_'.join(id_fields)
            full_id = full_id.replace(' ', '_')
        return full_id

    def get_values_from_field_list(self, field_list: list):
        output = []
        for attribute_name in field_list:
            single_output = self
            for attribute_part in attribute_name.split('.'):
                if hasattr(single_output, attribute_part):
                    single_output = getattr(single_output, attribute_part)
                else:
                    single_output = None
                    break
            if single_output:
                output.append(single_output)
        return output

    def get_custom_id_with_version_id(self, field_list: list):
        id_fields = self.get_values_from_field_list(field_list)
        id_fields.append( self.version_number)
        id_fields = [convert_datetime_to_string_utc(x) if isinstance(x, datetime) else x for x in id_fields]
        id_fields = [isodate.duration_isoformat(x) if isinstance(x, timedelta) else x for x in id_fields]
        id_fields = [x.value if isinstance(x, Enum) else x for x in id_fields]
        id_fields = [str(id_field) for id_field in id_fields if id_field]
        full_id = '_'.join(id_fields)
        full_id = full_id.replace(' ', '_')
        return full_id

    def generate_timeseries_id(self):
        pass

    def generate_id(self):
        id_fields = [self.generate_timeseries_id(), self.available.position]
        id_fields = [str(id_field) for id_field in id_fields if id_field]
        full_id = '_'.join(id_fields)
        full_id = full_id.replace(' ', '_')
        return full_id

    def generate_product_id(self):
        pass

    def get_xml_point(self):
        return  DataToXMLPoint(position=self.available.position,
                               primary_quantity=self.available.value,
                               secondary_quantity=self.percentage_level.value)


@dataclass
class CalculationPointBid(CalculationPoint):
    domain: Domain = None
    direction: DataPointType = None  # Downward

    def __post_init__(self):
        super().__post_init__()
        self.domain = check_dict_to_dataclass(Domain, self.domain)
        self.direction = check_dict_to_dataclass(DataPointType, self.direction)
        self.result_type = self.result_type or ProcurementCalculationType.NCPB.value

    def generate_timeseries_id(self):
        return  self.get_custom_id_with_version_id(field_list=BID_TIMESERIES_KEYS)

    def generate_product_id(self):
        return self.get_custom_id_with_version_id(field_list=BID_PRODUCT_KEYS)


@dataclass
class CalculationPointATC(CalculationPoint):
    in_domain: Domain = None
    out_domain: Domain = None

    def __post_init__(self):
        super().__post_init__()
        self.in_domain = check_dict_to_dataclass(Domain, self.in_domain)
        self.out_domain = check_dict_to_dataclass(Domain, self.out_domain)
        self.result_type = self.result_type or ProcurementCalculationType.ATC.value

    def generate_timeseries_id(self):
        return self.get_custom_id_with_version_id(field_list=ATC_TIMESERIES_KEYS)

    def generate_product_id(self):
        return self.get_custom_id_with_version_id(field_list=ATC_PRODUCT_KEYS)
