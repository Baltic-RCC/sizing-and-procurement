import math
from datetime import datetime, timedelta
from typing import Any

import pandas
import pytz

from py.common.time_functions import convert_datetime_to_string, convert_datetime_to_string_utc, time_delta_to_str
from py.data_classes.elastic.elastic_data_models import DataPointType, DataPoint, CalculationPointBid, \
    CalculationPointATC, QuantileSpacing
from py.data_classes.enums import FlowDirectionType, ProcurementCalculationType, BusinessProductType, \
    get_enum_name_value
from py.data_classes.task_classes import BusinessCode, EICArea, QuantileArray, MessageCommunicator, Domain
from py.procurement.constants import QUANTILES_INDEX_NAME

ESCAPE_NAN = True


def get_business_type_from_code(search_word: str, business_codes: list = None, business_type: str = None):
    """
    For parsing string to specific business code

    :param search_word: category name
    :param business_codes: list of business codes
    :param business_type: existing business_type if exists
    :return: business_type
    """
    if not business_type and business_codes:
        b_code = next(iter(b_type for b_type in business_codes
                           if b_type.business_type_product.name.lower() in search_word.lower()))
        if b_code and isinstance(b_code, BusinessCode):
            return str(b_code.business_type.value)
    return business_type


def get_domain_match_list(input_list: list, area_codes:list[EICArea] = None):
    """
    Gets domain and its sequence number from input list if exists

    :param input_list: list where to search (keys)
    :param area_codes: list of areas
    :return: Domain instance, sequence number of key if found, None, 0 otherwise
    """
    if area_codes is not None:
        for i, key_val in enumerate(input_list):
            match = next(iter(x for x in area_codes if x.value_of(key_val)), None)
            if match is not None:
                return match.get_domain(), i
    return None, 0


def get_match_from_dict(input_dict: dict, key_word: str, selection: Any):
    """
    Gets domain and its sequence number from input list if exists

    :param input_dict: dict where to search (keys)
    :param key_word: keyword to search
    :param selection: list of areas
    :return: Domain instance, sequence number of key if found, None, 0 otherwise
    """
    response = None
    key_value = next(iter([v for k, v in input_dict.items() if str(key_word).lower() in str(k).lower()]), None)
    if key_value is not None:
        selection = [selection] if not isinstance(selection, list) else selection
        try:
            match = next(iter([x for x in selection if x.value_of(key_value)]), None)
            response = match if match is not None else response
        except (ValueError, AttributeError):
            pass
    return response


def generate_bids_objects(data_to_send: pandas.DataFrame,
                          data_start_time: str | datetime,
                          data_end_time: str | datetime,
                          valid_from: str | datetime,
                          valid_to: str | datetime,
                          valid_resolution: str = None,
                          calculation_time: str | datetime = None,
                          data_resolution: str = 'PT15M',
                          main_unit: str = 'MW',
                          point_number: int = 1,
                          point_resolution: str | timedelta = None,
                          number_of_points: int = 1,
                          lfc_block: EICArea = None,
                          curve_type: str = 'A01',
                          version_number: int = 1,
                          secondary_unit: str = None,
                          product: str = None,
                          business_type: str = None,
                          business_types: list = None,
                          message_type: str = None,
                          process_type: str = None,
                          description: str = None,
                          sender: Domain | MessageCommunicator = None,
                          quantile_data: QuantileArray | QuantileSpacing = None,
                          index_name: str = QUANTILES_INDEX_NAME,
                          area_codes: list[EICArea] = None):
    """
    Main function to parse calculation results (quantiles) into bid object list

    :param sender: sender of calculation results
    :param valid_resolution: resolution between start and end time
    :param point_resolution: resolution of calculation result
    :param point_number: sequence number of calculation result within valid_resolution
    :param number_of_points: number of points in period
    :param data_to_send: calculation results
    :param data_start_time: start time from where the data was taken
    :param data_end_time: end time to where the data was taken
    :param valid_from: start time to which the results are valid
    :param valid_to: end time to which the results are valid
    :param calculation_time: timestamp at which the calculation was performed
    :param data_resolution: input data resolution
    :param main_unit: unit for the input data
    :param lfc_block: LFC block (code) in or for which the calculation was performed
    :param curve_type: if representing result type
    :param version_number: Version id for the calculation
    :param secondary_unit: measurement unit for sequence number (percentage)
    :param product: product id
    :param business_type: business type id
    :param business_types: available business types
    :param message_type: message type id
    :param process_type: process type id
    :param quantile_data: calculation results
    :param description: additional field to pass data
    :param index_name: y-axis in calculation results
    :param area_codes: available areas for getting the domain
    :return: list of CalculationPointBid instances
    """
    point_resolution = time_delta_to_str(point_resolution)
    valid_resolution = time_delta_to_str(valid_resolution)
    lfc_block = lfc_block.dict()
    output_data = data_to_send.reset_index().to_dict(orient='records')
    output_values = []
    calculation_time = calculation_time or convert_datetime_to_string(datetime.now(pytz.UTC))
    data_start_time = convert_datetime_to_string_utc(data_start_time)
    data_end_time = convert_datetime_to_string_utc(data_end_time)
    quantile_spacing = QuantileSpacing.from_spacing(spacing=quantile_data) \
        if isinstance(quantile_data, QuantileArray) else quantile_data

    for single_example in output_data:
        index_key = [key for key in single_example for match in key if index_name in match]
        index_key = index_key[0] if len(index_key) == 1 else None
        percentage_value = single_example.pop(index_key) if index_key is not None else "NaN"

        for data_key, data_value in single_example.items():
            if math.isnan(data_value) and ESCAPE_NAN:
                continue
            if len(data_key) >= 3:
                dir_name, dir_type = get_enum_name_value(input_value=data_key, enum_type=FlowDirectionType)
                direction = DataPointType(name=dir_name, code=dir_type)
                type_name, _ = get_enum_name_value(input_value=data_key, enum_type=BusinessProductType)
                data_point_type = DataPointType(name=type_name)
                domain, _ = get_domain_match_list(input_list=data_key, area_codes=area_codes)
                new_business_type = get_business_type_from_code(business_type=business_type,
                                                                business_codes=business_types,
                                                                search_word=type_name)
                available = DataPoint(position=point_number, value=data_value, measurement_unit=main_unit)
                percentile = DataPoint(position=point_number, value=percentage_value, measurement_unit=secondary_unit)
                output_values.append(CalculationPointBid(result_type=ProcurementCalculationType.NCPB.value,
                                                         data_resolution=data_resolution,
                                                         data_period_start=data_start_time,
                                                         data_period_end=data_end_time,
                                                         calculation_date=calculation_time,
                                                         valid_from=valid_from,
                                                         valid_to=valid_to,
                                                         valid_resolution=valid_resolution,
                                                         point_resolution=point_resolution,
                                                         number_of_points=number_of_points,
                                                         version_number=str(version_number),
                                                         available=available,
                                                         percentage_level=percentile,
                                                         LFC_block=lfc_block,
                                                         domain=domain,
                                                         direction=direction,
                                                         type=data_point_type,
                                                         quantile_spacing=quantile_spacing,
                                                         process_type=process_type,
                                                         message_type=message_type,
                                                         business_type=new_business_type,
                                                         curve_type=curve_type,
                                                         sender=sender,
                                                         description=description,
                                                         product=product))
    return output_values


def generate_atc_objects(data_to_send: pandas.DataFrame,
                         data_start_time: str | datetime,
                         data_end_time: str | datetime,
                         valid_from: str | datetime,
                         valid_to: str | datetime,
                         valid_resolution: str = None,
                         calculation_time: str | datetime = None,
                         data_resolution: str = 'PT15M',
                         main_unit: str = 'MW',
                         point_number: int = 1,
                         point_resolution: str | timedelta = None,
                         number_of_points: int = 1,
                         lfc_block: EICArea = None,
                         curve_type: str = 'A01',
                         version_number: int = 1,
                         secondary_unit: str = None,
                         product: str = None,
                         business_type: str = None,
                         business_types: list = None,
                         message_type: str = None,
                         process_type: str = None,
                         description: str = None,
                         sender: Domain | MessageCommunicator = None,
                         quantile_data: QuantileArray = None,
                         index_name: str = QUANTILES_INDEX_NAME,
                         area_codes: list[EICArea] = None):
    """
    Main function to parse calculation results (quantiles) into ATC object list

    :param sender: sender of calculation results
    :param valid_resolution: resolution between start and end time
    :param point_resolution: resolution of calculation result
    :param point_number: sequence number of calculation result within valid_resolution
    :param number_of_points: number of points in period
    :param data_to_send: calculation results
    :param data_start_time: start time from where the data was taken
    :param data_end_time: end time to where the data was taken
    :param valid_from: start time to which the results are valid
    :param valid_to: end time to which the results are valid
    :param calculation_time: timestamp at which the calculation was performed
    :param data_resolution: input data resolution
    :param main_unit: unit for the input data
    :param lfc_block: LFC block (code) in or for which the calculation was performed
    :param curve_type: id representing result type
    :param version_number: Version id for the calculation
    :param secondary_unit: measurement unit for sequence number (percentage)
    :param product: product id
    :param business_type: business type id
    :param business_types: available business types
    :param message_type: message type id
    :param process_type: process type id
    :param quantile_data: calculation results
    :param description: additional field to pass data
    :param index_name: y-axis in calculation results
    :param area_codes: available areas for getting the domain
    :return: list of CalculationPointATC instances
    """
    point_resolution = time_delta_to_str(point_resolution)
    valid_resolution = time_delta_to_str(valid_resolution)
    lfc_block = lfc_block.dict()
    business_type = get_business_type_from_code(business_type=business_type,
                                                business_codes=business_types,
                                                search_word='ATC')
    output_data = data_to_send.reset_index().to_dict(orient='records')
    index_names = list(data_to_send.columns.names)
    output_values = []
    calculation_time = calculation_time or convert_datetime_to_string(datetime.now(pytz.UTC))
    data_start_time = convert_datetime_to_string_utc(data_start_time)
    data_end_time = convert_datetime_to_string_utc(data_end_time)
    type_val = DataPointType(name=BusinessProductType.ATC.name)
    quantile_spacing = QuantileSpacing.from_spacing(spacing=quantile_data) \
        if isinstance(quantile_data, QuantileArray) else quantile_data
    for single_example in output_data:
        index_key = [key for key in single_example for match in key if index_name in match]
        index_key = index_key[0] if len(index_key) == 1 else None
        percentage_value = single_example.pop(index_key) if index_key is not None else "NaN"

        for data_key, data_value in single_example.items():
            if math.isnan(data_value) and ESCAPE_NAN:
                continue
            if len(data_key) >= 2:
                out_domain, seq_no = get_domain_match_list(input_list=data_key, area_codes=area_codes)
                in_domain, _ = get_domain_match_list(input_list=data_key[seq_no + 1::], area_codes=area_codes)
                if isinstance(index_names, list) and len(index_names) == len(data_key):
                    mapping = dict(zip(index_names, data_key))
                    in_v = get_match_from_dict(input_dict=mapping, key_word='in_domain', selection=area_codes)
                    out_v = get_match_from_dict(input_dict=mapping, key_word='out_domain', selection=area_codes)
                    out_domain = out_v.get_domain() if out_v and callable(hasattr(out_v, 'get_domain')) else out_domain
                    in_domain = in_v.get_domain() if in_v and callable(hasattr(in_v, 'get_domain')) else in_domain
                available = DataPoint(position=point_number, value=data_value, measurement_unit=main_unit)
                percentile = DataPoint(position=point_number,value=percentage_value, measurement_unit=secondary_unit)
                output_values.append(CalculationPointATC(result_type=ProcurementCalculationType.ATC.value,
                                                         data_resolution=data_resolution,
                                                         data_period_start=data_start_time,
                                                         data_period_end=data_end_time,
                                                         calculation_date=calculation_time,
                                                         valid_from=valid_from,
                                                         valid_to=valid_to,
                                                         type=type_val,
                                                         valid_resolution=valid_resolution,
                                                         point_resolution=point_resolution,
                                                         number_of_points=number_of_points,
                                                         version_number=str(version_number),
                                                         available=available,
                                                         percentage_level=percentile,
                                                         LFC_block=lfc_block,
                                                         in_domain=in_domain,
                                                         out_domain=out_domain,
                                                         quantile_spacing=quantile_spacing,
                                                         process_type=process_type,
                                                         message_type=message_type,
                                                         business_type=business_type,
                                                         curve_type=curve_type,
                                                         sender=sender,
                                                         description=description,
                                                         product=product))
    return output_values
