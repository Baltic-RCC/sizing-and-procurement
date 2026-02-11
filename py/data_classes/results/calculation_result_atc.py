import logging
import math
import os
from dataclasses import dataclass

import pandas

from py.common.functions import get_file_path_from_root_by_name, dict_to_dataclass
from py.common.time_functions import convert_datetime_to_string, time_delta_to_str
from py.common.ref_constants import AVAILABLE_VALUE_KEY, VALID_FROM_KEY, VALID_TO_KEY, PERCENTAGE_VALUE_KEY, \
    OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY
from py.data_classes.elastic.elastic_data_models import CalculationPointATC, QuantileSpacing
from py.data_classes.enums import ProcurementCalculationType
from py.data_classes.results.calculation_result_main import CalculationResult, NO_OF_LEVELS
from py.common.df_functions import add_levels_to_index
from py.data_classes.results.result_functions import DEFAULT_POINT_RESOLUTION, ATC_XML_DOCUMENT_TYPES, \
    check_resolutions, validate_xml
from py.data_classes.task_classes import TimeSliceResult
from py.procurement.constants import METADATA_KEY, DEFAULT_AREA
from py.handlers.rabbit_handler import R_SENDER_KEY, R_RECEIVER_KEY
from py.procurement.procurement_result_to_dto import generate_atc_objects

logger = logging.getLogger(__name__)

@dataclass
class CalculationResultATC(CalculationResult):
    calculation_type = ProcurementCalculationType.ATC

    def set_type(self):
        """
        Sets type

        :return:
        """
        self.calculation_type = ProcurementCalculationType.ATC

    def set_domains(self, domain_list: list = None):
        """
        Sets domains from area list.

        :param domain_list: Specify area list, otherwise the one from constants is taken
        :return:
        """
        domain_list = domain_list or DEFAULT_AREA
        for z in self.object_list:
            for x in z:
                x.in_domain = next(iter(y for y in domain_list if y.value_of(x.in_domain.mRID)), x.in_domain)
                x.out_domain = next(iter(y for y in domain_list if y.value_of(x.out_domain.mRID)), x.out_domain)

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
                                  columns=[OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY],
                                  aggfunc='min').reset_index()
        pivoted_columns = pivoted_table.columns.to_list()
        if no_of_levels is not None and no_of_levels > 0:
            delta = no_of_levels - next(iter(len(x) for x in pivoted_columns))
            pivoted_table = add_levels_to_index(input_df=pivoted_table, no_of_levels=delta)
        return pivoted_table

    def parse_dict_object_list(self, input_data: dict):
        """
        Generates list of CalculationPointATC instances input data

        :param input_data: nested dictionaries
        :return:
        """
        self._object_list = [[dict_to_dataclass(CalculationPointATC, data_dict) for data_dict in input_data]]

    def generate_object_list(self):
        """
        Generates list of CalculationPointATC instances from fields

        :return:
        """
        self._object_list = []
        self._version_numbers = []
        for quantile in self.quantiles:
            point_number = 1
            valid_from = self.valid_from
            valid_to = self.valid_to
            point_resolution = DEFAULT_POINT_RESOLUTION
            valid_resolution = None
            number_of_points = 1
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
                                                                      point_resolution=point_resolution)
                version_number = max(version_number, 1) + 1 if version_number else 1
                self._version_numbers.append(version_number)
            else:
                version_number = self.version
            self._object_list.append(generate_atc_objects(data_to_send=quantile.quantile_result,
                                                          data_start_time=self.data_period_start,
                                                          data_end_time=self.data_period_end,
                                                          valid_from=valid_from,
                                                          valid_to=valid_to,
                                                          valid_resolution=valid_resolution,
                                                          calculation_time=self.calculation_date,
                                                          data_resolution=self.time_resolution,
                                                          main_unit=str(self.main_unit.value),
                                                          point_number=point_number,
                                                          point_resolution=point_resolution,
                                                          number_of_points=number_of_points,
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
        self.calculation_type = self.calculation_type or ProcurementCalculationType.ATC
        try:
            doc_type = str(self.calculation_type.name)
        except AttributeError:
            doc_type = 'Missing'
        regrouped_elements = self.group_objects_to_timeseries()
        for file_type in ATC_XML_DOCUMENT_TYPES:
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
                    main_capacity, _ = xml_document.generate(receiver=receiver,
                                                             sender=self.sender,
                                                             doc_start_time=self.valid_from,
                                                             doc_end_time=self.valid_to,
                                                             mrid=mrid,
                                                             process_type=str(self.process_type.value),
                                                             message_type=str(self.message_type.value),
                                                             revision_number=str(version_value),
                                                             domain_mrid=self.lfc_block.mRID)
                    for atc_values in object_element.values():
                        atc = atc_values[0]
                        description = (f"{doc_type}_From_{atc.out_domain.name}_to_{atc.in_domain.name}_"
                                       f"at_{atc.percentage_level.value}")
                        values = [x.get_xml_point()  for x in atc_values if not math.isnan(x.available.value)]
                        if len(values) > 0:
                            point_resolution, values = check_resolutions(receiver=receiver,
                                                                         values=values,
                                                                         point_resolution=atc.point_resolution)
                            point_resolution = time_delta_to_str(point_resolution)
                            atc_series = xml_series.generate(series_start_time=atc.series_valid_from,
                                                             series_end_time=atc.series_valid_to,
                                                             resolution=point_resolution,
                                                             business_type=atc.business_type,
                                                             mrid=atc.mrid,
                                                             auction_mrid = description,
                                                             in_domain=atc.in_domain,
                                                             out_domain=atc.out_domain,
                                                             measurement_unit=atc.available.measurement_unit,
                                                             sec_measurement_unit=atc.percentage_level.measurement_unit,
                                                             curve_type=atc.curve_type,
                                                             product=atc.product,
                                                             values=values)
                            main_capacity.add_timeseries(atc_series)
                    validated = True
                    if not bypass_validation:
                        validated = validate_xml(main_capacity.to_xml().encode('utf-8'), xml_schema)
                    if validated:
                        xml_object = self.xml_to_bytes_io(xml_document=main_capacity,
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


if __name__ == '__main__':
    capacity_schema = r'../../resources/schemas/iec62325-451-3-capacity_v8_3.xsd'
    level_up = os.path.join(os.getcwd().split('py')[0], os.pardir)
    capacity_path = get_file_path_from_root_by_name(file_name=os.path.basename(capacity_schema), root_folder=level_up)

    print("Done")