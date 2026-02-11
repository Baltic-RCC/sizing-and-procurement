import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from typing import Any

import pandas

from py.common.functions import calculate_start_and_end_date, dict_to_dataclass
from py.common.time_functions import convert_string_to_datetime, parse_duration
from py.data_classes.enums import ProcurementCalculationType, BusinessType, ProcessType, MessageType, \
    MeasurementUnitType, CurveType, EnergyProductType, CodingSchemeType, StatusType, NameValueOfEnum, RoleType, \
    EICCodeType, BusinessProductType, FlowDirectionType
from py.common.ref_constants import VALID_FROM_KEY, VALID_TO_KEY, QUANTILE_START_KEY, QUANTILE_STEP_KEY, \
    QUANTILE_STOP_KEY
from py.handlers.elastic_handler import dict_to_and_or_query, BoolQueryKey

logger = logging.getLogger(__name__)

TIME_KEYS = [VALID_FROM_KEY, VALID_TO_KEY]
QUANTILE_KEYS = [QUANTILE_START_KEY, QUANTILE_STEP_KEY, QUANTILE_STOP_KEY]


def match_by_keys(old_ob, new_ob, primary_key: str | list = None):
    """
    Compares two objects by the keys given

    :param old_ob: object 1
    :param new_ob: object 2
    :param primary_key: attribute names
    :return: True if objects are matching by keys. False otherwise
    """
    if isinstance(primary_key, str):
        primary_key = [primary_key]
    for key in primary_key:
        if not (hasattr(old_ob, key) and hasattr(new_ob, key) and getattr(old_ob, key) == getattr(new_ob, key)):
            return False
    return True


def merge_object_lists_by_keys(old_list, new_list, primary_key: str | list):
    """
    Adds elements from the new_list to old_list if there is none based on primary_key

    :param old_list: existing list of objects
    :param new_list: list of objects to be added
    :param primary_key: keys by which to compare the objects
    :return: updated list
    """
    if not isinstance(new_list, list) or len(new_list) == 0:
        return old_list
    for new_ob in new_list:
        if not any(iter([x for x in old_list if match_by_keys(old_ob=x, new_ob=new_ob, primary_key=primary_key)])):
            old_list.append(new_ob)
    return old_list


def update_object_by_other(old_ob, new_ob, primary_key: str | list = None, mismatch_return_new: bool = True):
    """
    Updates old_ob by new_ob. if primary_key is given then updates only if matching

    :param old_ob: object to be updated
    :param new_ob: object that updates
    :param primary_key: attributes for equality of objects
    :param mismatch_return_new: if true returns new object if mismatch
    :return: updated or new object
    """
    same_types = True
    if primary_key is not None:
        same_types = match_by_keys(old_ob=old_ob, new_ob=new_ob, primary_key=primary_key)
    if same_types:
        for new_key, new_value in vars(new_ob).items():
            if new_value is not None:
                setattr(old_ob, new_key, new_value)
    elif mismatch_return_new:
        return new_ob
    return old_ob


def update_list_of_objects(old_list, new_list, primary_key: str | list, add_if_not_found: bool = True):
    """
    Updates one list of object by other list of objects

    :param old_list: list of objects to be updated
    :param new_list: list of object for updating
    :param primary_key: attribute names for matching
    :param add_if_not_found: if true then adds new object to list if no old matching one is found
    :return: updated list
    """
    if not isinstance(new_list, list) or len(new_list) == 0:
        return old_list
    for new_ob in new_list:
        replaced = False
        for place, old_ob in enumerate(old_list):
            if match_by_keys(old_ob=old_ob, new_ob=new_ob, primary_key=primary_key):
                old_list[place] = update_object_by_other(old_ob=old_ob, new_ob=new_ob, primary_key=primary_key)
                replaced = True
                break
        if not replaced and add_if_not_found:
            old_list.append(new_ob)
    return old_list


@dataclass
class Domain:
    name: str = None
    mRID: str = None
    market_role_type: str | RoleType = None

    @property
    def value(self):
        return self.mRID

    def value_of(self, search_str):
        """
        Checks whether the name of mRID corresponds to the search string

        :param search_str: value to search
        :return: self if value found None otherwise
        """
        if search_str is None:
            return None
        search_str = str(search_str).upper()
        mrid_check = str(self.mRID).upper() == search_str
        name_check = False
        if self.name is not None:
            self_name = str(self.name).upper()
            name_check = self_name in search_str
            # name_check = search_str == self_name
        if name_check or mrid_check:
            return self
        return None

    def dict(self):
        _dict = self.__dict__.copy()
        for k, v in _dict.items():
            if isinstance(v, Enum):
                _dict[k] = str(_dict[k])
        return _dict

    def __eq__(self, other):
        return isinstance(other, Domain) and self.mRID == other.mRID and self.name == other.name


@dataclass
class EICCodeOwner(Domain):
    mRID: str
    function: EICCodeType | str = None

    def get_domain(self):
        return Domain(name=self.name, mRID=self.mRID)

@dataclass
class EICParty(EICCodeOwner):

    def get_domain(self):
        role = self.market_role_type.value if isinstance(self.market_role_type, RoleType) else self.market_role_type
        return Domain(name=self.name, mRID=self.mRID, market_role_type=role)

@dataclass
class MessageCommunicator(EICParty):
    receiver: str = None
    output_resolution: str = None

    def __post_init__(self):
        if self.function is not None and isinstance(self.function, str):
            self.function = EICCodeType.value_of(self.function)
        if self.market_role_type is not None and isinstance(self.market_role_type, str):
            self.market_role_type = RoleType.value_of(self.market_role_type)

    def value_of(self, search_str):
        response = super().value_of(search_str)
        if not response and self.receiver:
            if str(search_str).upper() == str(self.receiver).upper():
                response = self
        return response

    def __eq__(self, other):
        return  (isinstance(other, MessageCommunicator) and
                 self.mRID == other.mRID and
                 self.name == other.name  and
                 self.receiver == other.receiver)

@dataclass
class EICArea(EICCodeOwner):
    area_code: str = None
    party: EICParty = None

    @staticmethod
    def init_from_dict(in_v: dict):
        mrid = in_v.get(next(iter(key for key in in_v if ('area' in key.lower() and 'eic' in key.lower())), None))
        name = in_v.get(next(iter(key for key in in_v if ('area' in key.lower() and 'name' in key.lower())), None))
        code = in_v.get(next(iter(key for key in in_v if ('area' in key.lower() and 'code' in key.lower())), None))
        x_id = in_v.get(next(iter(key for key in in_v if ('party' in key.lower() and 'eic' in key.lower())), None))
        x_name = in_v.get(next(iter(key for key in in_v if ('party' in key.lower() and 'name' in key.lower())), None))
        party = None
        if x_id or x_name:
            party = EICParty(mRID=x_id, name=x_name, function=EICCodeType.TSO)
        new_area = EICArea(mRID=mrid, name=name, area_code=code, function=EICCodeType.LFC_AREA, party=party)
        return new_area

    def value_of(self, search_str):
        """
        Checks whether any important field corresponds to the search string
        :param search_str: value to search
        :return: self if value found None otherwise
        """
        response = super().value_of(search_str)
        if not response and self.party and isinstance(self.party, EICParty):
            response = self.party.value_of(search_str)
        if not response and self.area_code:
            if str(search_str).upper() == str(self.area_code).upper():
                response = self
        return response


@dataclass
class NCBPCorrectionKey:
    country: str
    direction: FlowDirectionType
    borders: list

    def get_by_country_direction(self, inputs, areas: list[Domain]):
        """
        Checks country and direction fields from inputs
        :param inputs: list of strings
        :param areas: list of area instances
        :return: self if match found None otherwise
        """
        c_true = any(iter(y for x in areas for y in inputs if x.value_of(y) and x.value_of(self.country)))
        d_true = any(iter(y for y in inputs if str(self.direction.name).lower() in str(y).lower()))
        if c_true and d_true:
            return self
        return None


@dataclass
class TimeSlice:
    valid_from: str | datetime
    valid_to: str | datetime
    start_time: str | time = None
    end_time: str | time = None
    valid_period_resolution: str | timedelta = None
    point_resolution: str | timedelta = None
    number_of_points: int = 1
    point: int = 1
    time_zone: str = None

    def set_point_resolution(self, point_resolution: str = None):
        point_resolution = point_resolution or self.point_resolution
        self.point_resolution = parse_duration(point_resolution) \
            if isinstance(point_resolution, str) else point_resolution

    def init_values(self):
        self.valid_from = convert_string_to_datetime(self.valid_from)
        self.valid_to = convert_string_to_datetime(self.valid_to)
        if not self.time_zone:
            self.time_zone = self.valid_from.tzname()
        self.set_point_resolution()

    def __post_init__(self):
        self.init_values()
        self.get_point_resolution()
        self.get_period_resolution()
        self.get_start_end_time()

    def __eq__(self, other):
        if not isinstance(other, TimeSlice):
            return  NotImplemented
        is_same = True
        is_same =  False if self.valid_to != other.valid_to else is_same
        is_same =  False if self.valid_period_resolution != other.valid_period_resolution else is_same
        is_same =  False if self.point_resolution != other.point_resolution else is_same
        is_same =  False if self.number_of_points != other.number_of_points else is_same
        is_same =  False if self.point != other.point else is_same
        return is_same

    def get_slice_valid_from(self):
        return self.valid_from

    def get_slice_valid_to(self):
        return self.valid_to

    def get_calc_obj_valid_form(self):
        self.get_point_resolution()
        return self.valid_from - (self.point - 1) * self.point_resolution

    def get_calc_obj_valid_to(self):
        self.get_point_resolution()
        return self.get_calc_obj_valid_form() + self.number_of_points * self.point_resolution

    def get_point_resolution(self):
        if isinstance(self.point_resolution, str):
            self.point_resolution = parse_duration(self.point_resolution)
        if self.point_resolution is None:
            self.init_values()
            self.point_resolution = self.valid_to - self.valid_from
        return self.point_resolution

    def get_period_resolution(self):
        if isinstance(self.valid_period_resolution, str):
            self.valid_period_resolution = parse_duration(self.valid_period_resolution)
        if self.valid_period_resolution is None:
            valid_from = self.get_calc_obj_valid_form()
            valid_to = self.get_calc_obj_valid_to()
            self.valid_period_resolution = valid_to - valid_from
        return self.valid_period_resolution

    def get_start_end_time(self):
        self.start_time = self.start_time or self.get_slice_valid_from().time()
        self.end_time = self.end_time or self.get_slice_valid_to().time()

    def get_start_end_time_string(self, output_format: str = '%H%M'):
        self.get_start_end_time()
        return f"{self.start_time.strftime(output_format)}_{self.end_time.strftime(output_format)}"


@dataclass
class QuantileArray:
    spacing_start_value: float = None
    spacing_end_value: float = None
    spacing_step_size: float = None
    rounding: int = 4
    spacing: list = None

    def __eq__(self, other):
        if not isinstance(other, QuantileArray):
            return  NotImplemented
        is_same = True
        is_same =  False if self.spacing_start_value != other.spacing_start_value else is_same
        is_same =  False if self.spacing_end_value != other.spacing_end_value else is_same
        is_same =  False if self.spacing_step_size != other.spacing_step_size else is_same
        return is_same


    def __post_init__(self):
        """

        :return:
        """
        if self.spacing_start_value and self.spacing_end_value and self.spacing_step_size:
            self.calculate_spacing()

    def calculate_spacing(self):
        """
        Calculates spacing vector for quantiles
        """
        if any(x is None for x in [self.spacing_start_value, self.spacing_end_value, self.spacing_step_size]):
            return
        try:
            calculated_step_size = round(self.spacing_step_size / 100, self.rounding)
            start_value = round(1 - self.spacing_start_value / 100, self.rounding)
            end_value = round(1 - self.spacing_end_value / 100, self.rounding)
            spacing = [end_value]
            number_of_steps = round((self.spacing_end_value - self.spacing_start_value) / calculated_step_size) + 1
            default_end_value = 0
            for step in range(number_of_steps):
                new_step = round(default_end_value + step * calculated_step_size, self.rounding)
                if end_value < new_step <= start_value:
                    spacing.append(new_step)
            self.spacing = spacing
        except ValueError:
            pass


@dataclass
class QuantileResult:
    calculation_type: ProcurementCalculationType
    quantile_array: QuantileArray = field(default_factory=lambda: QuantileArray)
    quantile_result: pandas.DataFrame = field(default_factory=lambda: pandas.DataFrame())

    def same_timeframe(self, other):
        if not isinstance(other, QuantileResult):
            return NotImplemented
        return False if self.quantile_array != other.quantile_array else False

    def __eq__(self, other):
        is_same = self.same_timeframe(other)
        is_same = False if self.calculation_type != other.calculation_type else is_same
        return is_same


@dataclass
class TimeSliceResult(QuantileResult):
    time_slice: TimeSlice = None

    def same_time_slice(self, other):
        if not isinstance(other, TimeSliceResult):
            return NotImplemented
        return self.time_slice == other.time_slice

    def same_timeframe(self, other):
        if not isinstance(other, TimeSliceResult):
            return NotImplemented
        is_same = True
        is_same = False if self.time_slice != other.time_slice else is_same
        is_same = False if self.quantile_array != other.quantile_array else is_same
        return is_same

    @staticmethod
    def from_quantile_result(time_slice_result: TimeSlice, quantile_result: QuantileResult):
        return TimeSliceResult(time_slice=time_slice_result,
                               quantile_array=quantile_result.quantile_array,
                               calculation_type=quantile_result.calculation_type,
                               quantile_result=quantile_result.quantile_result)

    def get_quantile_result_time(self, valid_from: str = VALID_FROM_KEY, valid_to: str = VALID_TO_KEY):
        """
        Generates version for Excel where are the valid_from and valid_to columns as well
        :param valid_from: valid_from column name
        :param valid_to: valid_to column name
        :return: updated quantile_result
        """
        new_result =copy.deepcopy(self.quantile_result)
        new_result_columns = self.quantile_result.columns.to_list()
        index_depth = max([len(x) if isinstance(x, tuple) else 0 for x in new_result_columns])
        if index_depth > 0:
            valid_from = tuple([valid_from] + ['' for _ in (range(index_depth - 1))])
            valid_to = tuple([valid_to] + ['' for _ in (range(index_depth - 1))])
        new_result.loc[:, valid_from] = self.time_slice.get_slice_valid_from()
        new_result.loc[:, valid_to] = self.time_slice.get_slice_valid_to()
        new_result_columns = [valid_from, valid_to] + new_result_columns
        return new_result[new_result_columns]


@dataclass
class BusinessCode:
    business_type: str | BusinessType = None
    business_type_product: str | BusinessProductType = None

    def __post_init__(self):
        self.from_strings(business_type=self.business_type,
                          business_type_product=self.business_type_product)

    def from_strings(self, business_type: str = None, business_type_product: str = None):
        if business_type and isinstance(business_type, str):
            self.business_type = BusinessType.value_of(business_type)
        if business_type_product and isinstance(business_type_product, str):
            self.business_type_product = BusinessProductType.value_of(business_type_product)


@dataclass
class TypeCodes:
    business_types: dict | list = None
    process_type: str | ProcessType = None
    message_type: str | MessageType = None

    def __post_init__(self):
        self.from_strings(process_type=self.process_type,
                          message_type=self.message_type)
        self.business_types_from_dict(business_types=self.business_types)

    def from_strings(self, process_type: str = None, message_type: str = None):
        if process_type and isinstance(process_type, str):
            self.process_type = ProcessType.value_of(process_type)
        if message_type and isinstance(message_type, str):
            self.message_type = MessageType.value_of(message_type)

    def business_types_from_dict(self, business_types: dict | list):
        if isinstance(business_types, dict):
            self.business_types = [BusinessCode(business_type_product=key, business_type=value)
                                   for key, value in business_types.items()]


    @staticmethod
    def type_code_from_dict(input_dict):
        return dict_to_dataclass(TypeCodes, input_dict)


def set_enum(input_value, default_input_value = None, type_enum = NameValueOfEnum):
    """

    :param input_value:
    :param default_input_value:
    :param type_enum:
    :return:
    """
    if isinstance(input_value, NameValueOfEnum):
        return input_value
    try:
        return type_enum.value_of(input_value)
    except ValueError:
        return default_input_value or input_value


@dataclass
class ProcurementCalculationTask:
    calculation_type: ProcurementCalculationType

    valid_period_offset: str = '-P2D'
    data_period_timedelta: str = 'P60D'
    data_period_offset: str = None
    valid_period_timedelta: str = '-P1D'
    calculation_time_zone: str = 'UTZ'

    data_period_start: str | datetime = None
    data_period_end: str | datetime = None
    valid_from: str | datetime = None
    valid_to: str | datetime = None
    period_valid_time_shift: str = None
    sender: MessageCommunicator = None
    receivers: list[MessageCommunicator] = None
    atc_code_types: TypeCodes = None
    bids_code_types: TypeCodes = None
    power_unit_type: str | MeasurementUnitType = MeasurementUnitType.MEGAWATT
    percent_unit_type: str | MeasurementUnitType = MeasurementUnitType.PERCENT
    curve_type: str | CurveType = CurveType.POINT
    energy_product: str | EnergyProductType = EnergyProductType.ACTIVE_ENERGY
    coding_scheme: str | CodingSchemeType = CodingSchemeType.EIC
    status_type: str | StatusType = StatusType.AVAILABLE
    spacings: list[QuantileArray] = None
    time_slices: list[TimeSlice] = None
    lfc_areas: list[EICArea] = None
    lfc_block: EICArea = None
    description: str = None

    def __post_init__(self):
        """
        Default method for dataclass to run after initialization
        :return:
        """
        self.calculate_dates()
        self.set_default_enums()

    def set_default_enums(self,
                          power_type: MeasurementUnitType = MeasurementUnitType.MEGAWATT,
                          percentile_type: MeasurementUnitType = MeasurementUnitType.PERCENT,
                          curve_type: CurveType = CurveType.POINT,
                          energy_product: EnergyProductType = EnergyProductType.ACTIVE_ENERGY,
                          coding_scheme: CodingSchemeType = CodingSchemeType.EIC,
                          status_type: StatusType = StatusType.AVAILABLE,
                          ):
        """

        :param power_type:
        :param percentile_type:
        :param curve_type:
        :param energy_product:
        :param coding_scheme:
        :param status_type:
        :return:
        """
        self.power_unit_type = set_enum(self.power_unit_type, power_type, MeasurementUnitType)
        self.percent_unit_type = set_enum(self.percent_unit_type, percentile_type, MeasurementUnitType)
        self.curve_type = set_enum(self.curve_type, curve_type, CurveType)
        self.energy_product = set_enum(self.energy_product, energy_product, EnergyProductType)
        self.coding_scheme = set_enum(self.coding_scheme, coding_scheme, CodingSchemeType)
        self.status_type = set_enum(self.status_type, status_type, StatusType)



    def calculate_dates(self):
        """
        So, the logic how to calculate the valid time period and the period from which the results are valid
        The key is self.period_valid_time_shift.
        If this is determined then both period are coupled (this delta being the difference)
        if not then valid_period is whatever it is defined
        and data_period is ending with yesterday
        :return:
        """
        if not self.valid_to or not self.valid_from:
            valid_periods = calculate_start_and_end_date(start_date_time=self.valid_from,
                                                         end_date_time=self.valid_to,
                                                         offset=self.valid_period_offset,
                                                         time_zone=self.calculation_time_zone,
                                                         time_delta=self.valid_period_timedelta)
            self.valid_to, self.valid_from = valid_periods
        if not self.data_period_start or not self.data_period_end:
            data_offset = self.data_period_offset or self.period_valid_time_shift
            start_time = self.data_period_start
            end_time = self.data_period_end
            if self.period_valid_time_shift:
                start_time = start_time or self.valid_from
                end_time = end_time or self.valid_to
            data_periods = calculate_start_and_end_date(start_date_time=start_time,
                                                        end_date_time=end_time,
                                                        offset=data_offset,
                                                        time_delta=self.data_period_timedelta,
                                                        default_timedelta='P60D')
            self.data_period_start, self.data_period_end = data_periods


def get_domain_query(domain_value: Domain, keyword: str = 'domain'):
    """
    For adding domain fields to query

    :param domain_value: Domain instance
    :param keyword: keyword for distinguishing domains
    :return: query
    """
    query_dict = {f"{keyword}.mRID": domain_value.mRID, f"{keyword}.name": domain_value.name}
    return dict_to_and_or_query(value_dict=query_dict,
                                bool_key=BoolQueryKey.SHOULD)


@dataclass
class Operator:
    method: Any = None
    parameters: dict = None
