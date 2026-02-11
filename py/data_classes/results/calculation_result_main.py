import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any

import pandas
import pytz

from py.common.functions import get_random_string_length, align_country_names, get_value_by_position
from py.common.df_functions import parse_dataframe_to_nested_dict, remove_empty_levels_from_index, \
    rename_multilevel_index_levels
from py.common.time_functions import convert_datetime_to_string, convert_datetime_to_string_utc, str_to_datetime, \
    time_delta_to_str, get_datetime_columns_of_data_frame
from py.data_classes.elastic.elastic_data_models import CalculationPoint, QuantileSpacing
from py.data_classes.enums import ProcurementCalculationType, ProcessType, MessageType, MeasurementUnitType, CurveType, \
    CodingSchemeType, EnergyProductType, StatusType, parse_to_enum
from py.data_classes.results.result_functions import get_value_from_column, DEFAULT_POINT_RESOLUTION, get_version_number_from_elastic, \
    handle_dataframe_timezone_excel
from py.common.ref_constants import PROCESS_TYPE_KEY, VERSION_KEY, MESSAGE_TYPE_KEY, CURVE_TYPE_KEY, PERCENTAGE_UNIT_KEY, \
    AVAILABLE_UNIT_KEY, VALID_FROM_KEY, VALID_TO_KEY, PRODUCT_KEY, SENDER_MRID_KEY, PERCENTAGE_VALUE_KEY, \
    QUANTILE_START_KEY, QUANTILE_STEP_KEY, QUANTILE_STOP_KEY
from py.data_classes.task_classes import QuantileResult, ProcurementCalculationTask, EICArea, MessageCommunicator, \
    Domain, TimeSliceResult, TimeSlice, QuantileArray, TIME_KEYS, update_list_of_objects, BusinessCode, get_domain_query
from py.handlers.elastic_handler import PY_ELASTICSEARCH_HOST, factory, send_dictionaries_to_elastic, \
    nested_dict_to_flat, PY_PROCUREMENT_PROPOSED_INDEX, dict_to_and_or_query
from py.procurement.constants import PY_VERSION_NUMBER, QUANTILES_INDEX_NAME, PY_CALCULATION_STEPS, \
    PY_TABLE_COUNTRY_KEY, FINAL_SENDER, FINAL_RECEIVER, AVAILABLE_SENDERS, DEFAULT_AREA, PERCENTAGE_MEASUREMENT_UNIT, \
    CURVE_TYPE, POWER_MEASUREMENT_UNIT, ENERGY_PRODUCT

NO_OF_LEVELS = 3


def pivoted_table_for_excel(input_df, area_codes: list = None, country_code:str = PY_TABLE_COUNTRY_KEY):
    """
    Additional function to prepare pivoted dataframe for the Excel

    :param input_df: input area codes
    :param area_codes: countries if naming is needed
    :param country_code: naming key
    :return: updated dataframe
    """
    area_codes = area_codes or DEFAULT_AREA
    input_df = align_country_names(input_dataframe=input_df, area_list=area_codes, attribute_name=country_code)
    input_df = rename_multilevel_index_levels(input_df, {'mRID': country_code})
    input_df= handle_dataframe_timezone_excel(input_data=input_df).reset_index()
    columns = get_datetime_columns_of_data_frame(input_data=input_df)
    if columns is not None and len(columns) >= 1:
        for col_name in columns:
            input_df[col_name] = input_df[col_name].apply(lambda x: convert_datetime_to_string_utc(x))

    return input_df


def generate_excel_sheet_name(fields: list, delimiter='_', random_str_len: int = 1):
    """
    Generates sheet names from the values given if not None. Adds additional random string to the end and cuts it
    into max allowed length (31)

    :param fields: list of fields for sheet name
    :param random_str_len: length of random string
    :param delimiter: delimiter string
    :return: sheet name as string
    """
    default_len = 31
    component_length = default_len - (random_str_len + len(delimiter)) if random_str_len > 0 else default_len
    random_str = get_random_string_length(k=random_str_len) if random_str_len > 0 else None
    sheet_name_components = [str(x) for x in fields if x is not None]
    sheet_name = delimiter.join(sheet_name_components)
    sheet_name = sheet_name[:component_length]
    sheet_name = delimiter.join([x for x in [sheet_name, random_str] if x is not None])
    return sheet_name[:default_len]


def get_senders_from_dataframe(input_data: pandas.DataFrame,
                               sender_list: list = None,
                               sender_column_name: str = SENDER_MRID_KEY):
    """
    Maps the sender column to MessageCommunicator class. Updates the fields with hardcoded values if applicable

    :param input_data: input dataframe
    :param sender_list: if provided uses this to update found sender objects. if not then reverts to default ones
    :param sender_column_name: column containing sender information
    :return: list of senders
    """
    sender_list = sender_list or [*FINAL_SENDER, *FINAL_RECEIVER]
    if sender_column_name in input_data.columns:
        senders = input_data[sender_column_name].unique().tolist()
        sender_objects = [MessageCommunicator(mRID=x) for x in senders]
        updated_senders = update_list_of_objects(old_list=sender_objects,
                                                 new_list=sender_list,
                                                 primary_key='mRID',
                                                 add_if_not_found=False)
        return updated_senders
    return None


def set_param(old: Any = None, new: Any = None, overwrite: bool = True):
    return new if overwrite else old or new


@dataclass
class CalculationResult:

    input_data: pandas.DataFrame = field(default_factory=lambda: pandas.DataFrame)
    pivoted_data: pandas.DataFrame = field(default_factory=lambda: pandas.DataFrame)
    quantiles: list[QuantileResult] = field(default_factory=lambda: list)
    task: ProcurementCalculationTask = None
    calculation_type: ProcurementCalculationType = None
    version: int = PY_VERSION_NUMBER
    percentage_index_column: str | tuple = QUANTILES_INDEX_NAME
    elastic_server: str = PY_ELASTICSEARCH_HOST
    elastic_index: str = PY_PROCUREMENT_PROPOSED_INDEX
    area_codes: list[EICArea] = None
    _object_list: list = None
    _version_numbers: list = None

    # Initiated from task
    data_period_start: str | datetime = None
    data_period_end: str | datetime = None
    valid_from: str | datetime = None
    valid_to: str | datetime = None
    process_type: str | ProcessType = None
    message_type: str | MessageType = None
    business_types: str | list = None
    sender: MessageCommunicator = None
    receivers: list[MessageCommunicator] = field(default_factory=lambda: list)
    main_unit: str | MeasurementUnitType = None
    secondary_unit: str | MeasurementUnitType = None
    curve_type: str | CurveType = None
    coding_scheme: str | CodingSchemeType = None
    energy_product: str | EnergyProductType = None
    status_type: str | StatusType = None
    lfc_block: Domain | EICArea = None
    # Initiated as is
    calculation_date: str | datetime = None
    # Initiated from dataframe
    time_resolution: str = None

    excel_sheet_prefix: str = None
    description: str = None

    def init_from_task(self, task: ProcurementCalculationTask = None, overwrite: bool = True):
        """
        If task is given, packs its values to fields

        :param overwrite:
        :param task: ProcurementCalculationTask instance
        :return: None
        """
        task = task or self.task
        if task is not None:
            self.valid_from = set_param(old=self.valid_from, new=task.valid_from, overwrite=overwrite)
            self.valid_to = set_param(old=self.valid_to, new=task.valid_to, overwrite=overwrite)
            self.main_unit = set_param(old=self.main_unit, new=task.power_unit_type, overwrite=overwrite)
            self.secondary_unit = set_param(old=self.secondary_unit, new=task.percent_unit_type, overwrite=overwrite)
            self.curve_type = set_param(old=self.curve_type, new=task.curve_type, overwrite=overwrite)
            self.energy_product = set_param(old=self.energy_product, new=task.energy_product, overwrite=overwrite)
            self.coding_scheme = set_param(old=self.coding_scheme, new=task.coding_scheme, overwrite=overwrite)
            self.status_type = set_param(old=self.status_type, new=task.status_type, overwrite=overwrite)
            self.data_period_end = set_param(old=self.data_period_start, new=task.data_period_end, overwrite=overwrite)
            self.data_period_start = set_param(old=self.data_period_start, new=task.data_period_start,
                                               overwrite=overwrite)
            self.description = set_param(old=self.description, new=task.description, overwrite=overwrite)
            process_type = None
            message_type = None
            business_types = None
            if self.calculation_type == ProcurementCalculationType.NCPB:
                process_type = task.bids_code_types.process_type
                message_type = task.bids_code_types.message_type
                business_types = task.bids_code_types.business_types
            elif self.calculation_type == ProcurementCalculationType.ATC:
                process_type = task.atc_code_types.process_type
                message_type = task.atc_code_types.message_type
                business_types = task.atc_code_types.business_types

            self.process_type = set_param(old=self.process_type, new=process_type, overwrite=overwrite)
            self.message_type = set_param(old=self.message_type, new=message_type, overwrite=overwrite)
            self.business_types = set_param(old=self.business_types, new=business_types, overwrite=overwrite)

            self.sender = set_param(old=self.sender, new=task.sender, overwrite=overwrite)
            self.receivers = set_param(old=self.receivers, new=task.receivers, overwrite=overwrite)
            self.lfc_block = set_param(old=self.lfc_block, new=task.lfc_block, overwrite=overwrite)
            self.area_codes = set_param(old=self.area_codes, new=task.lfc_areas, overwrite=overwrite)

    def set_type(self):
        pass

    def __post_init__(self):
        """
        Post-init processes
        Fills in the fields

        :return: None
        """
        self.set_type()
        if not self.calculation_date:
            self.calculation_date = datetime.now(pytz.utc)
        if not self.input_data.empty:
            self.lfc_block = self.lfc_block or EICArea(mRID = get_value_from_column(input_data=self.input_data,
                                                                                    column_string='lfc_block.mrid'))
            self.time_resolution = self.time_resolution or get_value_from_column(input_data=self.input_data,
                                                                                 column_string='resolution')
        self.init_from_task()

    def set_object_list(self, object_list: list):
        """
        Setter for setting the object list

        :param object_list:  CalculationPoint instances
        :return: None
        """
        self._object_list = object_list

    def parse_dict_object_list(self, input_data: dict):
        """
        Interface to extend based on subclass

        :param input_data: nested dictionaries (usually got from dataframe)
        :return:
        """
        pass

    def update_sender(self, sender: Domain | MessageCommunicator = None):
        """
        Update sender for the object_list

        :param sender: new sender if specified, if not default one is used
        :return: None
        """
        if sender is not None:
            self.sender = sender
        if self.sender is not None:
            if isinstance(self.sender, MessageCommunicator):
                sender_domain = self.sender.get_domain()
            else:
                sender_domain = self.sender
            for single_list in self.object_list:
                for single_value in single_list:
                    single_value.sender = sender_domain

    def object_list_to_dataframe(self):
        """
        Generates dataframe from self object_list

        :return: dataframe from the object fields
        """
        outputs = []
        inputs = self.object_list
        if isinstance(inputs, list) and len(inputs) > 0:
            for object_element in inputs:
                outputs.extend(nested_dict_to_flat(asdict(element, dict_factory=factory)) for element in object_element)
        objects_df = pandas.DataFrame(outputs)
        return objects_df

    def update_quantiles_from_object_list(self):
        """
        Generates quantiles from object list

        :return:
        """
        self._get_quantiles_from_dataframe(input_data=self.object_list_to_dataframe())

    def update_object_list_from_quantiles(self, quantiles):
        self.quantiles = quantiles
        self.generate_object_list()

    def get_pivot_table(self, input_data: pandas.DataFrame):
        """
        Parses input data to pivot list (needed for Excel tables)

        :param input_data: input dataframe
        :return: pivoted table
        """
        return pandas.DataFrame()

    def _pivot_table(self, input_data: pandas.DataFrame):
        """
        Parses input data to pivot list (needed for excel tables)

        :param input_data: input dataframe
        :return:
        """
        self.pivoted_data = self.get_pivot_table(input_data=input_data)

    def _get_quantiles_from_dataframe(self, input_data: pandas.DataFrame):
        """
        Parses input data to for quantiles (needed for excel tables)

        :param input_data: input dataframe
        :return:
        """
        time_columns = TIME_KEYS
        self.quantiles = []
        quantile_columns = [x for x in input_data.columns.to_list() if 'quantile' in x.lower()]
        group_columns = time_columns + quantile_columns
        groups = input_data.groupby(group_columns, dropna=False)
        areas = self.task.lfc_areas if self.task else None
        for group_name, group_value in groups:
            name_tuple = tuple(group_name)
            if len(name_tuple) == len(group_columns):
                from_v = get_value_by_position(key_value=VALID_FROM_KEY,
                                               key_list=group_columns, value_list=group_name)
                to_v = get_value_by_position(key_value=VALID_TO_KEY,
                                             key_list=group_columns, value_list=group_name)
                q_start = get_value_by_position(key_value=QUANTILE_START_KEY,
                                                key_list=group_columns, value_list=group_name)
                q_step = get_value_by_position(key_value=QUANTILE_STEP_KEY,
                                               key_list=group_columns, value_list=group_name)
                q_stop = get_value_by_position(key_value=QUANTILE_STOP_KEY,
                                               key_list=group_columns, value_list=group_name)
                new_time_slice = TimeSlice(valid_from=from_v, valid_to=to_v)
                quantile_array = QuantileArray(spacing_start_value=q_start, spacing_end_value=q_step,
                                               spacing_step_size=q_stop)
                pivoted_group = self.get_pivot_table(input_data=group_value)
                time_col_vals = [x for x in pivoted_group.columns.to_list() for y in time_columns if y in x]
                percentage_col = [x for x in pivoted_group.columns.to_list() if PERCENTAGE_VALUE_KEY in x]
                pivoted_group = pivoted_group.drop(columns=time_col_vals).set_index(percentage_col)
                if isinstance(areas, list):
                    pivoted_group = align_country_names(input_dataframe=pivoted_group,
                                                        area_list=areas,
                                                        attribute_name=PY_TABLE_COUNTRY_KEY)
                new_quantile = TimeSliceResult(time_slice=new_time_slice,
                                               quantile_array=quantile_array,
                                               calculation_type=self.calculation_type,
                                               quantile_result=pivoted_group)
                self.quantiles.append(new_quantile)

    def preprocess_dataframe(self, input_data: pandas.DataFrame):
        """
        Adds additional filters, modifications to input dataframe if needed

        :param input_data: input data
        :return:
        """
        return input_data

    def set_domains(self, domain_list: list = None):
        """
        Sets domains from area list.

        :param domain_list: Specify area list, otherwise the one from constants is taken
        :return:
        """
        pass

    def dataframe_to_object_list(self,
                                 input_data: pandas.DataFrame,
                                 column_mapping: dict = None,
                                 valid_from: str | datetime = None,
                                 valid_to: str | datetime = None,
                                 message_type: str | MessageType = None,
                                 process_type: str | ProcessType = None,
                                 main_unit: str = POWER_MEASUREMENT_UNIT,
                                 percentage_unit: str = PERCENTAGE_MEASUREMENT_UNIT,
                                 energy_product: str = ENERGY_PRODUCT,
                                 curve_type: str = CURVE_TYPE,
                                 version_number: str | int = None):
        """
        For parsing dataframe to internal fields

        :param energy_product: specify energy product code if needed
        :param curve_type: to specify curve type code if needed
        :param percentage_unit: to specify percentage unit if needed
        :param main_unit: to specify main unit if needed
        :param input_data: to be input dataframe (single level index)
        :param column_mapping: map column names corresponding to fields of CalculationPoint if needed
        :param valid_from: start date of data, if not specified min from valid_from is taken
        :param valid_to: end date of data, if not specified max from valid_to is taken
        :param message_type: Message type value, if not specified first value from "message_type" column is taken
        :param process_type: process type value, if not specified first value from "process_type" column is taken
        :param version_number: version number, if not specified max value from "version_number" is taken
        :return: None
        """
        if column_mapping is not None:
            input_data = input_data.rename(columns=column_mapping)
        input_data = self.preprocess_dataframe(input_data=input_data)
        input_data = str_to_datetime(data=input_data, columns=TIME_KEYS)
        self._pivot_table(input_data=input_data)
        self._get_quantiles_from_dataframe(input_data=input_data)
        if valid_from:
            input_data[VALID_FROM_KEY] = valid_from
        if valid_to:
            input_data[VALID_TO_KEY] = valid_to
        if process_type:
            input_data[PROCESS_TYPE_KEY] = process_type.value if isinstance(process_type, ProcessType) else process_type
        if message_type:
            input_data[MESSAGE_TYPE_KEY] = message_type.value if isinstance(message_type, MessageType) else message_type
        self.parse_dict_object_list(input_data=parse_dataframe_to_nested_dict(input_dataframe=input_data))

        self.valid_from = valid_from or input_data[VALID_FROM_KEY].min()
        self.valid_to = valid_to or input_data[VALID_TO_KEY].max()
        self.message_type = message_type or MessageType.value_of(get_value_from_column(input_data=input_data,
                                                                                       column_string=MESSAGE_TYPE_KEY))
        self.process_type = process_type or ProcessType.value_of(get_value_from_column(input_data=input_data,
                                                                                       column_string=PROCESS_TYPE_KEY))
        self.version = version_number or max(pandas.to_numeric(input_data[VERSION_KEY], errors='coerce'))
        sec_unit = get_value_from_column(input_data=input_data, column_string=PERCENTAGE_UNIT_KEY) or percentage_unit
        main_unit = get_value_from_column(input_data=input_data, column_string=AVAILABLE_UNIT_KEY) or main_unit
        curve = get_value_from_column(input_data=input_data, column_string=CURVE_TYPE_KEY) or curve_type
        product = get_value_from_column(input_data=input_data, column_string=PRODUCT_KEY) or energy_product
        self.main_unit = parse_to_enum(main_unit, MeasurementUnitType)
        self.secondary_unit = parse_to_enum(sec_unit, MeasurementUnitType)
        self.curve_type = parse_to_enum(curve, CurveType)
        self.energy_product = parse_to_enum(product, EnergyProductType)
        self.lfc_block = next(iter([x.LFC_block for y in self._object_list for x in y]))

        business_types = set([x.business_type for y in self._object_list for x in y])
        if len(business_types) > 0:
            new_business_types = []
            for bs in business_types:
                try:
                    new_business_types.append(BusinessCode(business_type=bs))
                except ValueError:
                    pass
            if len(new_business_types) > 0 and self.business_types is None:
                self.business_types = new_business_types
        sender = next(iter([x.sender for y in self._object_list for x in y]), None)
        if sender is not None:
            self.sender = next(iter([x for x in AVAILABLE_SENDERS if x.mRID == sender.mRID]), sender)
        receiver = get_value_from_column(input_data=input_data, column_string='receiver_marketParticipant.mRID')
        if receiver is not None:
            self.receivers = [x for x in AVAILABLE_SENDERS if x.mRID == receiver]
            if len(self.receivers) == 0:
                self.receivers = [MessageCommunicator(mRID=receiver)]
        self.set_domains()
        self.input_data = input_data

    def update_version_numbers(self, elastic_index):
        """
        Queries version numbers to data (divided by quantile parameters)

        :param elastic_index: index where results are stored
        :return: None
        """
        series_listed = self.group_objects_to_timeseries()
        for single_series in series_listed:
            first_item = next(iter([y for x in list(single_series.values()) for y in x]))
            version_number = self.get_version_number_for_quantile(quantile=first_item.quantile_spacing,
                                                                  elastic_index=elastic_index,
                                                                  point_resolution=first_item.point_resolution)
            if version_number is not None:
                new_version_number = version_number + 1
                for single_value in single_series.values():
                    for item in single_value:
                        item.version_number = str(new_version_number)

    @property
    def object_list(self):
        """
        Generates list of CalculationPoint instances from input data

        :return: list of CalculationPoint instances
        """
        if not self._object_list:
            self.generate_object_list()
        return self._object_list

    def generate_object_list(self):
        """
        Interface for generating object lists

        :return:
        """
        pass

    def send_results_to_elastic(self, elastic_server: str = None, elastic_index: str = None):
        """
        Sends results to elastic index (dictionary with id: value)

        :param elastic_server: address of elastic server
        :param elastic_index: index where to send the results
        :return: None
        """
        elastic_server = elastic_server or self.elastic_server
        elastic_index = elastic_index or self.elastic_index
        for object_element in self.object_list:
            payload = {element.generate_id(): asdict(element, dict_factory=factory) for element in object_element}
            send_dictionaries_to_elastic(input_list=payload,
                                         elastic_server=elastic_server,
                                         elastic_index_name=elastic_index)

    def generate_xml(self, bypass_validation: bool = False, **kwargs):
        """
        Interface for generating xml files

        :param bypass_validation: Set true if validation is needed to be skipped
        :param kwargs: additional arguments
        :return:
        """
        pass

    def generate_xlsx_sheets(self,
                             add_object_list: bool = False,
                             add_input_data: bool = False,
                             add_quantile_data: bool = False,
                             add_pivoted_data: bool = False):
        """
        Generates Excel sheets from input data

        :param add_quantile_data: Add pivoted data from quantiles
        :param add_object_list: parse object list to dataframe and add it to excel
        :param add_input_data: True: adds input data as DataFrame
        :param add_pivoted_data: True: adds pivoted data as DataFrame
        :return: dictionary with Excel sheet name: DataFrame
        """
        outputs = {}
        if add_input_data and not self.input_data.empty:
            input_sheet_name = generate_excel_sheet_name(fields=[self.excel_sheet_prefix,
                                                                 self.calculation_type.value,
                                                                 self.process_type.value,
                                                                 self.message_type.value,
                                                                 'input'])
            outputs[input_sheet_name] = handle_dataframe_timezone_excel(input_data=self.input_data)
        if add_pivoted_data and not self.pivoted_data.empty:
            pivot_sheet_name = generate_excel_sheet_name(fields=[self.excel_sheet_prefix,
                                                                 self.calculation_type.value,
                                                                 self.process_type.value,
                                                                 self.message_type.value,
                                                                 'pivoted'])
            outputs[pivot_sheet_name] = pivoted_table_for_excel(input_df=self.pivoted_data,
                                                                area_codes=self.area_codes)
        if add_object_list:
            object_sheet_name = generate_excel_sheet_name(fields=[self.excel_sheet_prefix,
                                                                  self.calculation_type.value,
                                                                  self.process_type.value,
                                                                  self.message_type.value,
                                                                 'values'])
            outputs[object_sheet_name] = handle_dataframe_timezone_excel(input_data=self.object_list_to_dataframe())
        if isinstance(self.quantiles, list) and len(self.quantiles) >= 1 and add_quantile_data:
            for single_quantile in self.quantiles:
                sheet_name_components = []
                if single_quantile.quantile_array is not None:
                    sheet_name_components = [self.excel_sheet_prefix,
                                             str(single_quantile.calculation_type.value),
                                             str(single_quantile.quantile_array.spacing_start_value),
                                             str(single_quantile.quantile_array.spacing_step_size),
                                             str(single_quantile.quantile_array.spacing_end_value)]
                if isinstance(single_quantile, TimeSliceResult):
                    sheet_name_components.append(single_quantile.time_slice.get_start_end_time_string())
                sheet_name = generate_excel_sheet_name(sheet_name_components)
                outputs[sheet_name] = pivoted_table_for_excel(input_df=single_quantile.get_quantile_result_time(),
                                                              area_codes=self.area_codes)
        outputs = {k: remove_empty_levels_from_index(v) for k, v in outputs.items()}
        return outputs

    def group_objects_to_timeseries(self):
        """
        Groups data to time series based on quantile values (start-step-stop)

        :return: nested list of CalculationPoint instances
        """
        # group by parameters
        parameters = {}
        for object_single in self.object_list:
            for point in object_single:
                if isinstance(point, CalculationPoint):
                    id_value = point.generate_spacing_id()
                    if not id_value in parameters.keys():
                        parameters[id_value] = []
                    parameters[id_value].append(point)
        parameter_values = list(parameters.values())
        time_series_list = []
        for parameter_value in parameter_values:
            timeseries_dict = {}
            for point in parameter_value:
                if isinstance(point, CalculationPoint):
                    id_value = point.generate_timeseries_id()
                    if not id_value in timeseries_dict.keys():
                        timeseries_dict[id_value] = []
                    timeseries_dict[id_value].append(point)
            time_series_list.append(timeseries_dict)
        return time_series_list

    def xml_to_bytes_io(self, xml_document,
                        prefixes: str | list = None,
                        file_type: str = None,
                        to_mrid: str = None,
                        delimiter: str = '-'):
        """
        Converts xml document string to BytesIO object and adds a name to it

        :param xml_document: xml document string
        :param prefixes: prefixes to add file name
        :param to_mrid: address where to send (for file name)
        :param file_type: file type
        :param delimiter: character to tie together components. EDX likes only '-'
        :return: BytesIO object
        """
        random_nr = 5
        report_instance = BytesIO((xml_document.to_xml()).encode('utf-8'))
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        components = [] if not prefixes else copy.deepcopy(prefixes)
        components.append(file_type)
        components.append(f"x{str(PY_CALCULATION_STEPS or 1)}")
        components.append(convert_datetime_to_string(self.calculation_date, output_format='%Y-%m-%dT%H-%MZ'))
        if to_mrid:
            components.extend(['for', to_mrid])
        components.append(f"{get_random_string_length(random_nr)}.xml")
        report_instance.name = (delimiter.join([str(x) for x in components if (x is not None and x != '')]))
        return report_instance

    def get_version_number_for_quantile(self,
                                        quantile: QuantileResult | QuantileSpacing,
                                        valid_from: str | datetime = None,
                                        valid_to: str | datetime = None,
                                        elastic_index: str = None,
                                        point_resolution: str | timedelta = None):
        """
        Gets version number based on the quantile values (start, step, stop values)

        :param valid_to: valid to for quantile
        :param valid_from:  valid from for quantile
        :param quantile: Instance containing start_percentage, step_percentage, stop_percentage
        :param elastic_index: index where data is
        :param point_resolution: data resolution
        :return: version number if found
        """
        elastic_index = elastic_index or self.elastic_index
        if point_resolution is None:
            point_resolution = quantile.time_slice.point_resolution \
                if isinstance(quantile, TimeSliceResult) else DEFAULT_POINT_RESOLUTION
        if isinstance(quantile, QuantileResult):
            quantile = QuantileSpacing.from_spacing(spacing=quantile.quantile_array)
        valid_from = valid_from or self.valid_from
        valid_to = valid_to or self.valid_to
        version_number = get_version_number_from_elastic(calculation_type=self.calculation_type,
                                                         point_resolution=point_resolution,
                                                         quantile_start_value=quantile.start_percentage,
                                                         quantile_step_value=quantile.step_percentage,
                                                         quantile_stop_value=quantile.stop_percentage,
                                                         valid_from=valid_from,
                                                         valid_to=valid_to,
                                                         results_index=elastic_index)
        return version_number

    def update_version_number_for_objects(self, index_name: str = None):
        """
        Use this to update realised data version numbers

        :param index_name: where to query data
        :return: None
        """
        index_name = index_name or self.elastic_index
        for ob_list in self._object_list:
            for single_obj in ob_list:
                q_object = single_obj.quantile_spacing
                start_x = getattr(single_obj, 'start_percentage') if hasattr(q_object, 'start_percentage') else None
                step_y = getattr(single_obj, 'step_percentage') if hasattr(q_object, 'step_percentage') else None
                stop_z = getattr(single_obj, 'stop_percentage') if hasattr(q_object, 'stop_percentage') else None
                resolution = time_delta_to_str(single_obj.point_resolution)
                query_dict = {'business_type': single_obj.business_type, SENDER_MRID_KEY: single_obj.sender.mRID}
                query_values = []
                if single_obj.result_type == ProcurementCalculationType.ATC.value:
                    query_values.append(get_domain_query(domain_value=single_obj.in_domain, keyword='in_domain'))
                    query_values.append(get_domain_query(domain_value=single_obj.out_domain, keyword='out_domain'))
                elif single_obj.result_type == ProcurementCalculationType.NCPB.value:
                    query_values.append(get_domain_query(domain_value=single_obj.domain))
                    query_dict['direction.code'] = single_obj.direction.code
                custom_query = dict_to_and_or_query(value_dict=query_dict, value_list=query_values)
                version_number = get_version_number_from_elastic(calculation_type=single_obj.result_type,
                                                                point_resolution= resolution,
                                                                quantile_start_value=start_x,
                                                                quantile_step_value=step_y,
                                                                quantile_stop_value=stop_z,
                                                                valid_from=single_obj.valid_from,
                                                                valid_to=single_obj.valid_to,
                                                                additional_elements=custom_query,
                                                                results_index=index_name)

                if version_number:
                    try:
                        version_number = int(version_number)
                        single_obj.version_number = str(max(int(single_obj.version_number), version_number + 1))
                    except ValueError:
                        pass


