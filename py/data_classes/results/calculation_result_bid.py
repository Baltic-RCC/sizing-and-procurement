import logging
import math
import os
from dataclasses import dataclass

import pandas

from py.common.functions import get_file_path_from_root_by_name, dict_to_dataclass, ordered_sublist
from py.common.time_functions import convert_datetime_to_string, time_delta_to_str
from py.common.ref_constants import AVAILABLE_VALUE_KEY, VALID_FROM_KEY, VALID_TO_KEY, PERCENTAGE_VALUE_KEY, \
    DOMAIN_MRID_KEY, TYPE_NAME_KEY, DIRECTION_NAME_KEY, BUSINESS_TYPE_KEY, DIRECTION_CODE_KEY
from py.data_classes.elastic.elastic_data_models import CalculationPointBid, QuantileSpacing
from py.data_classes.enums import ProcurementCalculationType, BusinessType, BusinessProductType, FlowDirectionType, \
    IndicatorType, get_enum_name_value
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_main import CalculationResult, NO_OF_LEVELS
from py.common.df_functions import add_levels_to_index, align_two_dataframe_levels
from py.data_classes.results.result_functions import DEFAULT_POINT_RESOLUTION, BID_XML_DOCUMENT_TYPES, \
    check_resolutions, validate_xml
from py.data_classes.task_classes import TimeSliceResult, NCBPCorrectionKey
from py.procurement.constants import METADATA_KEY, CORRECTION_KEYS, DEFAULT_AREA
from py.handlers.rabbit_handler import R_SENDER_KEY, R_RECEIVER_KEY
from py.procurement.procurement_result_to_dto import generate_bids_objects

logger = logging.getLogger(__name__)


def group_columns(ncpb_columns: list,
                  atc_columns: list,
                  suggestions: list[NCBPCorrectionKey] = None,
                  areas: list = None):
    """
    Groups columns by products

    :param ncpb_columns: list of NCPB quantile columns
    :param atc_columns: list ATC quantile columns
    :param suggestions: list of correction keys
    :param areas: list area instances
    :return: mapping dictionary in form of column to update: columns from where to calculate
    """
    suggestions = suggestions or CORRECTION_KEYS
    areas = areas or DEFAULT_AREA
    mapping_dict = {}
    for ncpb_col in ncpb_columns:
        sub_cols = [ncpb_col]
        ncpb_col = [ncpb_col] if isinstance(ncpb_col, str) else ncpb_col
        suggestion = next(iter(x for x in suggestions if x.get_by_country_direction(ncpb_col, areas)), None)
        if suggestion:
            for border_pair in suggestion.borders:
                for atc_col in atc_columns:
                    new_atc_col = [atc_col] if isinstance(atc_col, str) else atc_col
                    matches = [next(iter(x for x in new_atc_col for y in areas
                                         if (y.value_of(x) and y.value_of(z))), None)
                               for z in border_pair]
                    matches = [x for x in matches if x is not None]
                    if len(matches) > 1 and ordered_sublist(inputs=matches, lookup=new_atc_col):
                        sub_cols.append(atc_col)
        mapping_dict[ncpb_col] = sub_cols
    return mapping_dict


@dataclass
class CalculationResultBid(CalculationResult):
    NCPB_updated: bool = False
    calculation_type = ProcurementCalculationType.NCPB

    def set_type(self):
        """
        Sets type

        :return:
        """
        self.calculation_type = ProcurementCalculationType.NCPB

    def set_domains(self, domain_list: list = None):
        """
        Sets domains from area list.

        :param domain_list: Specify area list, otherwise the one from constants is taken
        :return:
        """
        domain_list = domain_list or DEFAULT_AREA
        for z in self.object_list:
            for x in z:
                x.domain = next(iter(y for y in domain_list if y.value_of(x.domain.mRID)), x.domain)


    def get_pivot_table(self, input_data: pandas.DataFrame):
        """
        Generates pivoted table from input data (needed for Excel)

        :param input_data: input dataframe
        :return: pivoted table
        """
        no_of_levels = NO_OF_LEVELS
        pivoted_table = pandas.pivot_table(input_data,
                                  values=AVAILABLE_VALUE_KEY,
                                  index=[VALID_FROM_KEY, VALID_TO_KEY, PERCENTAGE_VALUE_KEY],
                                  columns=[DOMAIN_MRID_KEY, TYPE_NAME_KEY, DIRECTION_NAME_KEY],
                                  aggfunc='min').reset_index()
        pivoted_columns = pivoted_table.columns.to_list()
        if no_of_levels is not None and no_of_levels > 0:
            delta = no_of_levels - next(iter(len(x) for x in pivoted_columns))
            pivoted_table = add_levels_to_index(input_df=pivoted_table, no_of_levels=delta)
        return pivoted_table

    def preprocess_dataframe(self, input_data: pandas.DataFrame):
        """
        Additional fixes for bid dataframe

        :param input_data:  converted dataframe (from xml)
        :return: updated dataframe
        """
        type_name_keys = {BusinessType.AUTOMATIC_FREQUENCY_RESTORATION_RESERVE: BusinessProductType.aFRR.name,
                          BusinessType.MANUAL_FREQUENCY_RESTORATION_RESERVE: BusinessProductType.mFRR.name}
        input_data.loc[:, TYPE_NAME_KEY] = (input_data[BUSINESS_TYPE_KEY]
                                          .apply(lambda x: type_name_keys.get(BusinessType.value_of(x))))
        input_data.loc[:, DIRECTION_NAME_KEY] = (input_data[DIRECTION_CODE_KEY]
                                               .apply(lambda x: get_enum_name_value(input_value=x,
                                                                                    enum_type=FlowDirectionType)[0]))
        return input_data

    def parse_dict_object_list(self, input_data: dict):
        """
        Generates list of CalculationPointBid instances input data

        :param input_data: nested dictionaries
        :return:
        """
        self._object_list = [[dict_to_dataclass(CalculationPointBid, data_dict) for data_dict in input_data]]

    def generate_object_list(self):
        """
        Generates list of CalculationPointBid instances from fields

        :return:
        """
        self._object_list = []
        self._version_numbers = []
        for quantile in self.quantiles:
            point_number = 1
            valid_from = self.valid_from
            valid_to = self.valid_to
            point_resolution = DEFAULT_POINT_RESOLUTION
            number_of_points = 1
            valid_resolution = None
            if isinstance(quantile, TimeSliceResult) and quantile.time_slice is not None:
                point_number = quantile.time_slice.point
                valid_from = quantile.time_slice.valid_from
                valid_to = quantile.time_slice.valid_to
                point_resolution = quantile.time_slice.point_resolution
                valid_resolution = quantile.time_slice.valid_period_resolution
                number_of_points = quantile.time_slice.number_of_points
            quantile_spacing = QuantileSpacing.from_spacing(spacing=quantile.quantile_array)
            if not self.version:
                version_number = self.get_version_number_for_quantile(quantile=quantile_spacing,
                                                                      valid_from=valid_from,
                                                                      valid_to=valid_to,
                                                                      point_resolution=point_resolution)
                version_number = max(version_number, 1) + 1 if version_number else 1
                self._version_numbers.append(version_number)
            else:
                version_number = self.version
            self._object_list.append(generate_bids_objects(data_to_send=quantile.quantile_result,
                                                           data_start_time=self.data_period_start,
                                                           data_end_time=self.data_period_end,
                                                           calculation_time=self.calculation_date,
                                                           valid_from=valid_from,
                                                           valid_to=valid_to,
                                                           valid_resolution=valid_resolution,
                                                           data_resolution=self.time_resolution,
                                                           main_unit=str(self.main_unit.value),
                                                           point_number=point_number,
                                                           number_of_points=number_of_points,
                                                           point_resolution=point_resolution,
                                                           lfc_block=self.lfc_block,
                                                           curve_type=str(self.curve_type.value),
                                                           version_number=int(version_number),
                                                           secondary_unit=str(self.secondary_unit.value),
                                                           product=str(self.energy_product.value),
                                                           business_types=self.business_types,
                                                           message_type=str(self.message_type.value),
                                                           process_type=str(self.process_type.value),
                                                           quantile_data=quantile_spacing,
                                                           index_name=self.percentage_index_column,
                                                           sender=self.sender,
                                                           description=self.description,
                                                           area_codes=self.area_codes))

    def generate_xml(self, bypass_validation: bool = False, additional_prefixes: str | list = None):
        """
        Generates xml from object_list

        :param bypass_validation: Set true if it is needed to skip valida
        :param additional_prefixes: for file name
        :return: list of xml documents
        """
        output_reports = []
        if self.sender is None or self.receivers is None or len(self.receivers) == 0:
            logger.warning(f"Missing either sender or receiver for generating xml, returning none")
            return output_reports
        divisible = str(IndicatorType.YES.value)
        self.calculation_type = self.calculation_type or ProcurementCalculationType.ATC
        doc_type = str(self.calculation_type.name)
        regrouped_elements = self.group_objects_to_timeseries()
        for file_type in BID_XML_DOCUMENT_TYPES:
            xml_document = file_type.value.document_name
            xml_series = file_type.value.series_name
            xml_schema = file_type.value.xsd_schema_path
            file_type_name = file_type.value.file_name_prefix
            version_value = self.version or 1
            if isinstance(self._version_numbers, list) and len(self._version_numbers) > 0:
                version_value = max(max(self._version_numbers), version_value)
            for object_element in regrouped_elements:
                for receiver in self.receivers:
                    doc_start_time = convert_datetime_to_string(self.valid_from, output_format='%Y-%m-%dT%H:%MZ')
                    doc_end_time = convert_datetime_to_string(self.valid_to, output_format='%Y-%m-%dT%H:%MZ')
                    mrid = f"{doc_type}_from_{doc_start_time}_to_{doc_end_time}_for_{self.lfc_block.name}"
                    main_bid, _ = xml_document.generate(receiver=receiver,
                                                        sender=self.sender,
                                                        doc_start_time=self.valid_from,
                                                        doc_end_time=self.valid_to,
                                                        mrid=mrid,
                                                        process_type=str(self.process_type.value),
                                                        message_type=str(self.message_type.value),
                                                        revision_number=str(version_value),
                                                        domain_mrid=self.lfc_block.mRID)
                    for bids in object_element.values():
                        bid = bids[0]
                        values = [x.get_xml_point()  for x in bids if not math.isnan(x.available.value)]
                        if len(values) > 0:
                            point_resolution, values = check_resolutions(receiver=receiver,
                                                                         values=values,
                                                                         point_resolution=bid.point_resolution)
                            description = f"{doc_type}_{bid.domain.name}_{bid.type.name}_{bid.direction.name}"
                            point_resolution = time_delta_to_str(point_resolution)
                            bid_series = xml_series.generate(series_start_time=bid.series_valid_from,
                                                             series_end_time=bid.series_valid_to,
                                                             resolution=point_resolution,
                                                             business_type=bid.business_type,
                                                             mrid=bid.mrid,
                                                             auction_mrid=description,
                                                             divisible=divisible,
                                                             acquiring_domain=bid.LFC_block,
                                                             connecting_domain=bid.domain,
                                                             measurement_unit=bid.available.measurement_unit,
                                                             sec_measurement_unit=bid.percentage_level.measurement_unit,
                                                             flow_direction=str(bid.direction.code),
                                                             curve_type=bid.curve_type,
                                                             values=values)
                            main_bid.add_timeseries(bid_series)
                    validated = True
                    if not bypass_validation:
                        validated = validate_xml(main_bid.to_xml().encode('utf-8'), xml_schema)
                    if validated:
                        xml_object = self.xml_to_bytes_io(xml_document=main_bid,
                                                          prefixes=additional_prefixes,
                                                          file_type=file_type_name,
                                                          to_mrid=receiver.mRID)
                        meta_data = {R_RECEIVER_KEY: receiver,
                                     R_SENDER_KEY: self.sender,
                                     'type': file_type_name,
                                     'start_time': doc_start_time,
                                     'end_time': doc_end_time}
                        setattr(xml_object, METADATA_KEY, meta_data)
                        output_reports.append(xml_object)
                    else:
                        logger.warning(f"Unable to create xml")
        return output_reports

    def update_ncpb_by_atc(self,
                           atc_results: CalculationResultATC,
                           suggestions: list = None,
                           areas: list = None):
        """
        Corrects NCPB data by the ATC results.
        Note that correction happens quantile-wise (precent against percent).
        For that valid_from, valid_to, resolution, step_start, step, step_end must match
        :param areas: list of country data
        :param suggestions: list min keys
        :param atc_results: CalculationResultATC instance
        :return:None
        """
        areas = areas or self.area_codes
        for ncpb_quantile in self.quantiles:
            for atc_quantile in atc_results.quantiles:
                if isinstance(atc_quantile, TimeSliceResult) and isinstance(ncpb_quantile, TimeSliceResult):
                    if ncpb_quantile.same_timeframe(atc_quantile):
                        ncpb_df, atc_df = align_two_dataframe_levels(input_df_1=ncpb_quantile.quantile_result,
                                                                     input_df_2=atc_quantile.quantile_result)
                        mapping_dict = group_columns(ncpb_columns=ncpb_df.columns.to_list(),
                                                          atc_columns=atc_df.columns.to_list(),
                                                          suggestions=suggestions,
                                                          areas= areas)
                        pre_levels = ncpb_df.columns.names
                        merged_data = ncpb_df.merge(atc_df, how='left',left_index=True,right_index=True)
                        for key_col in mapping_dict:
                            merged_data[key_col] = merged_data[mapping_dict[key_col]].min(axis=1)
                        merged_data = merged_data[ncpb_df.columns.to_list()]
                        merged_data.columns = merged_data.columns.set_names(list(pre_levels))
                        ncpb_quantile.quantile_result = merged_data
        self.NCPB_updated = True


if __name__ == '__main__':
    reserve_schema = r'../../resources/schemas/iec62325-451-7-reservebiddocument_v7_6.xsd'
    balancing_schema = r'../../resources/schemas/iec62325-451-6-balancing_v4_5.xsd'
    merit_schema = r'../../resources/schemas/iec62325-451-7-moldocument_v7_3.xsd'

    level_up = os.path.join(os.getcwd().split('py')[0], os.pardir)
    reserve_path = get_file_path_from_root_by_name(file_name=os.path.basename(reserve_schema), root_folder=level_up)
    balancing_path = get_file_path_from_root_by_name(file_name=os.path.basename(balancing_schema), root_folder=level_up)
    merit_path = get_file_path_from_root_by_name(file_name=os.path.basename(merit_schema), root_folder=level_up)

    print("Done")