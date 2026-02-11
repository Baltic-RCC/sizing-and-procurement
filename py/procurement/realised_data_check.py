import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy
import pandas
import pytz

from py.data_classes.elastic.elastic_data_models import CalculationPoint
from py.data_classes.enums import ProcurementCalculationType, ExceededPercentType, ExceededEnumOperator
from py.data_classes.results.calculation_result_main import get_senders_from_dataframe
from py.common.ref_constants import SENDER_MRID_KEY, RECEIVER_MRID_KEY, PERCENTAGE_VALUE_KEY, AVAILABLE_VALUE_KEY, \
    VALID_FROM_KEY, VALID_TO_KEY, POINT_RESOLUTION_KEY, BUSINESS_TYPE_KEY, OUT_DOMAIN_MRID_KEY, IN_DOMAIN_MRID_KEY, \
    DOMAIN_MRID_KEY, DIRECTION_CODE_KEY
from py.data_classes.task_classes import MessageCommunicator, get_domain_query, Operator
from py.handlers.elastic_handler import PY_PROCUREMENT_REALISED_INDEX, PY_PROCUREMENT_PROPOSED_INDEX, \
    dict_to_and_or_query, factory, nested_dict_to_flat, merge_queries, PY_PROCUREMENT_EXCEEDED_INDEX
from py.handlers.rabbit_handler import PY_RMQ_EXCHANGE
from py.parsers.json_to_calculation_result import get_calculation_results_from_elastic, UNIQUE_COLUMN_KEYWORDS, \
    delete_columns, filter_dataframe_to_latest_by_key, filter_calculation_results_by_type, \
    convert_to_calculation_result
from py.parsers.parser_constants import xml_sender, PY_OUTPUT_CUSTOM_QUERY
from py.parsers.xlsx_to_calculation_result import parse_excel_to_calculation_results, EXCEL_INCLUDED_SHEETS, \
    EXCEL_EXCLUDED_SHEETS
from py.procurement.constants import PY_EXCEEDED_SIGN, PY_OUTPUT_SEND_XML_TO_MINIO, \
    PY_OUTPUT_SEND_XML_TO_RABBIT, PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE, PY_EXCEEDED_RMQ_HEADERS, XML_FOLDER_TO_STORE, \
    PY_OUTPUT_SEND_XLSX_TO_MINIO, PY_OUTPUT_SEND_XLSX_TO_RABBIT, PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE, \
    EXCEL_FOLDER_TO_STORE, FINAL_SENDER, FINAL_RECEIVER, PY_EXCEEDED_NCPB_CODES, PY_EXCEEDED_ATC_CODES, \
    PY_EXCEEDED_PREFIX, PY_EXCEEDED_TYPE, PY_EXCEEDED_PERCENT, PY_EXCEEDED_REPORT, PY_EXCEEDED_INITIAL, \
    PY_EXCEEDED_OPERATOR
from py.procurement.procurement_common import get_task_from_environment
from py.common.time_functions import time_delta_to_str
from py.procurement.procurement_output import handle_parsed_output

logger = logging.getLogger(__name__)

PROP_SUFFIX = '_pre'
REAL_SUFFIX = '_post'
NORM_FACTOR = 'norm_constant'
EXCEEDED_KEY = 'exceeded'
TIMESTAMP_KEY = '@timestamp'
DEFAULT_COLUMNS = [VALID_FROM_KEY, VALID_TO_KEY, POINT_RESOLUTION_KEY, BUSINESS_TYPE_KEY]
ATC_COLUMNS = [IN_DOMAIN_MRID_KEY, OUT_DOMAIN_MRID_KEY]
NCPB_COLUMNS = [DOMAIN_MRID_KEY, DIRECTION_CODE_KEY]


@dataclass
class RealisedDataObject:
    realised_atc: pandas.DataFrame = field(default_factory=lambda: pandas.DataFrame)
    realised_ncpb: pandas.DataFrame = field(default_factory=lambda: pandas.DataFrame)
    receivers = None
    sender = None
    prefixes = PY_EXCEEDED_PREFIX

    def parse_input(self,
                    input_df: pandas.DataFrame,
                    report_type: ProcurementCalculationType = PY_EXCEEDED_TYPE,
                    atc_columns: list | str = None,
                    ncpb_columns: list | str = None,
                    key_column: str = TIMESTAMP_KEY):
        """
        Groups and reorganizes results by type and receiver. Note that multiple files may be received during the
        session. Therefore, filter to latest result if multiple values for TSO + product id are present

        :param report_type:
        :param input_df: input dataframe
        :param atc_columns: group atc products by these
        :param ncpb_columns: group ncpb products by these
        :param key_column: column for filtering out duplicates
        :return: None
        """
        atc_columns = atc_columns or [*DEFAULT_COLUMNS, *ATC_COLUMNS]
        ncpb_columns = ncpb_columns or [*DEFAULT_COLUMNS, *NCPB_COLUMNS]
        if input_df.empty:
            return
        if report_type == ProcurementCalculationType.NCPB or report_type == ProcurementCalculationType.ALL:
            self.realised_ncpb = filter_calculation_results_by_type(input_data=input_df,
                                                                    fixed_columns= ncpb_columns,
                                                                    key_column=key_column,
                                                                    result_type=ProcurementCalculationType.NCPB)
            self.realised_ncpb = self.realised_ncpb[self.realised_ncpb[AVAILABLE_VALUE_KEY].notna()]
        if report_type == ProcurementCalculationType.ATC or report_type == ProcurementCalculationType.ALL:
            self.realised_atc = filter_calculation_results_by_type(input_data=input_df,
                                                                   fixed_columns=atc_columns,
                                                                   key_column=key_column,
                                                                   result_type=ProcurementCalculationType.ATC)
            self.realised_atc = self.realised_atc[self.realised_atc[AVAILABLE_VALUE_KEY].notna()]
        exceeded_task = get_task_from_environment(ncpb_codes=PY_EXCEEDED_NCPB_CODES, atc_codes=PY_EXCEEDED_ATC_CODES)
        results = []
        if not self.realised_ncpb.empty:
            results.extend(convert_to_calculation_result(input_data=self.realised_ncpb,
                                                         download_type=ProcurementCalculationType.NCPB,
                                                         receivers=self.receivers,
                                                         task=exceeded_task,
                                                         process_type=exceeded_task.bids_code_types.process_type,
                                                         message_type=exceeded_task.bids_code_types.message_type,
                                                         sender=self.sender))
        if not self.realised_atc.empty:
            results.extend(convert_to_calculation_result(input_data=self.realised_atc,
                                                         download_type=ProcurementCalculationType.ATC,
                                                         receivers=self.receivers,
                                                         task=exceeded_task,
                                                         process_type=exceeded_task.atc_code_types.process_type,
                                                         message_type=exceeded_task.atc_code_types.message_type,
                                                         sender=self.sender))
        if len(results) > 0:
            # calc_date = datetime.now(pytz.utc)
            # for res in results:
            #     res.calculation_date = calc_date
            handle_parsed_output(results=results,
                                 output_to_elastic=True,
                                 elastic_index=PY_PROCUREMENT_EXCEEDED_INDEX,
                                 prefixes=self.prefixes,
                                 output_xml_minio=PY_OUTPUT_SEND_XML_TO_MINIO,
                                 output_xml_rabbit=PY_OUTPUT_SEND_XML_TO_RABBIT,
                                 output_xml_local=PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE,
                                 rabbit_exchange=PY_RMQ_EXCHANGE,
                                 rabbit_headers=PY_EXCEEDED_RMQ_HEADERS,
                                 xml_local_path=XML_FOLDER_TO_STORE,
                                 output_xlsx_minio=PY_OUTPUT_SEND_XLSX_TO_MINIO,
                                 output_xlsx_rabbit=PY_OUTPUT_SEND_XLSX_TO_RABBIT,
                                 output_xlsx_local=PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE,
                                 add_corrected_ncpb=False,
                                 xlsx_local_path=EXCEL_FOLDER_TO_STORE)


@dataclass
class RealisedDataCollector:
    buffer = []
    realised_atc = None
    realised_ncpb = None
    prefixes = PY_EXCEEDED_PREFIX
    sender = xml_sender

    def add_to_buffer(self, input_df):
        """
        Collects values to buffer

        :param input_df: dataframe of exceeded values
        :return: None
        """
        self.buffer.append(input_df)

    def parse_input(self,
                    receiver_list: list = None,
                    rounding_decimals: int = 1,
                    receiver_col: str = RECEIVER_MRID_KEY,
                    sender: MessageCommunicator |list = None):
        """
        Groups and reorganizes results by type and receiver

        :param receiver_list: list of TSOs
        :param rounding_decimals: for nicer look round the values (if -1 then no rounding happens)
        :param receiver_col: column name where to send
        :param sender: default sender
        :return: None
        """
        if len(self.buffer) > 0:
            logger.info(f"Sending out realised values")
            final_df = pandas.concat(self.buffer)
            if final_df.empty:
                return
            if rounding_decimals >= 0:
                final_df[AVAILABLE_VALUE_KEY] = (final_df[AVAILABLE_VALUE_KEY]
                                                 .apply(lambda x: round(x, rounding_decimals)))
            receiver_list = receiver_list or  [*FINAL_SENDER, *FINAL_RECEIVER]
            sender = sender or FINAL_SENDER
            sender = sender[0] if isinstance(sender, list) else sender
            df_receivers = get_senders_from_dataframe(input_data=final_df,
                                                      sender_list=receiver_list,
                                                      sender_column_name=receiver_col)
            for receiver in df_receivers:
                slice_df = final_df[final_df[receiver_col] == receiver.mRID]
                output_object = RealisedDataObject()
                output_object.sender = sender
                output_object.receivers = [receiver]
                output_object.parse_input(input_df=slice_df)


def get_data_for_realise_check(elastic_index: str,
                               result_type: ProcurementCalculationType,
                               valid_from: str | datetime = None,
                               valid_to: str | datetime = None,
                               point_resolution: str = None,
                               start_x: str | float = None,
                               step_y: str | float = None,
                               stop_z: str | float = None,
                               custom_query: dict = None,
                               key_words: list = None):
    """
    Queries data realised check

    :param elastic_index: index name for elastic
    :param result_type: result type (NCPB or ATC)
    :param valid_from: start time for the result
    :param valid_to: end time for the result
    :param point_resolution: resolution for the value
    :param start_x: quantile start percentage (90%)
    :param step_y: quantile step (0.01%)
    :param stop_z: quantile stop percentage (100%)
    :param custom_query: add additional parameters
    :param key_words: use these for filtering for the latest
    :return: dataframe of results
    """
    key_words = key_words or UNIQUE_COLUMN_KEYWORDS
    received = get_calculation_results_from_elastic(start_date_time=valid_from,
                                                    end_date_time=valid_to,
                                                    elastic_index=elastic_index,
                                                    download_type=result_type,
                                                    point_resolution=point_resolution,
                                                    start_x=start_x,
                                                    step_y=step_y,
                                                    stop_z=stop_z,
                                                    custom_query=custom_query)
    received = filter_dataframe_to_latest_by_key(input_dataframe=received,
                                                 variable_columns=key_words,
                                                 key_column='version')
    received = received.reset_index(drop=True)

    # Remove this in the future: just in case if there are multiple same values from same sender, go for the latest
    # entry
    if elastic_index == PY_PROCUREMENT_REALISED_INDEX and not received.empty:
        received = received.loc[received.groupby(SENDER_MRID_KEY)[TIMESTAMP_KEY].idxmax()]

    received = delete_columns(received, columns_delete=[TIMESTAMP_KEY], delete_empty=True)
    return received


def get_proposed_realised_for_calc_ob(calc_ob, key_words: list = None, additional_query: dict =PY_OUTPUT_CUSTOM_QUERY):
    """
    Queries proposed and realised values based on CalculationPoint data

    :param additional_query:
    :param calc_ob: CalculationPoint instance
    :param key_words: use these for filtering to latest
    :return: proposed dataframe and realised dataframe
    """
    key_words = key_words or UNIQUE_COLUMN_KEYWORDS
    realised_key_words = [*key_words, PERCENTAGE_VALUE_KEY] if PERCENTAGE_VALUE_KEY else key_words

    q_object = calc_ob.quantile_spacing
    start_x = getattr(q_object, 'start_percentage') if hasattr(q_object, 'start_percentage') else None
    step_y = getattr(q_object, 'step_percentage') if hasattr(q_object, 'step_percentage') else None
    stop_z = getattr(q_object, 'stop_percentage') if hasattr(q_object, 'stop_percentage') else None
    query_dict = {}
    query_values = []
    if calc_ob.result_type == ProcurementCalculationType.ATC.value:
        query_values.append(get_domain_query(domain_value=calc_ob.in_domain, keyword='in_domain'))
        query_values.append(get_domain_query(domain_value=calc_ob.out_domain, keyword='out_domain'))
    elif calc_ob.result_type == ProcurementCalculationType.NCPB.value:
        query_values.append(get_domain_query(domain_value=calc_ob.domain))
        query_dict[DIRECTION_CODE_KEY] = calc_ob.direction.code
        query_dict[BUSINESS_TYPE_KEY] = calc_ob.business_type
    custom_query = dict_to_and_or_query(value_dict=query_dict, value_list=query_values)
    custom_query = merge_queries(custom_query, additional_query)
    resolution = time_delta_to_str(calc_ob.point_resolution)
    realised = get_data_for_realise_check(elastic_index=PY_PROCUREMENT_REALISED_INDEX,
                                          result_type=calc_ob.result_type,
                                          valid_from=calc_ob.valid_from,
                                          valid_to=calc_ob.valid_to,
                                          point_resolution=resolution,
                                          start_x=start_x,
                                          step_y=step_y,
                                          stop_z=stop_z,
                                          key_words=key_words,
                                          custom_query=custom_query)
    proposed = get_data_for_realise_check(elastic_index=PY_PROCUREMENT_PROPOSED_INDEX,
                                          result_type=calc_ob.result_type,
                                          valid_from=calc_ob.valid_from,
                                          valid_to=calc_ob.valid_to,
                                          point_resolution=resolution,
                                          start_x=start_x,
                                          step_y=step_y,
                                          stop_z=stop_z,
                                          key_words=realised_key_words,
                                          custom_query=custom_query)
    return proposed, realised


def update_dataframe_from_calc_object(input_df: pandas.DataFrame, calc_obj: CalculationPoint, keys: list | str):
    """
    Updates dataframe based on CalculationPoint

    :param input_df: input dataframe
    :param calc_obj: CalculationPoint instance
    :param keys: common keys (columns for dataframe, attributes/columns for calculation point instance)
    :return: updated dataframe
    """
    response_dict = nested_dict_to_flat(input_dict=asdict(calc_obj, dict_factory=factory))
    response_df = pandas.DataFrame([response_dict])
    if not input_df.empty:
        input_df.set_index(keys, inplace=True)
        input_df.update(response_df.set_index(keys, inplace=True))
        return input_df.reset_index()
    return response_df


def get_obj_fields_from_dataframe(input_df: pandas.DataFrame,
                                  input_obj: dataclass,
                                  keys: list | str,
                                  fields: list | str = None):
    """


    :param input_df:
    :param input_obj:
    :param keys:
    :param fields:
    :return:
    """
    keys = [keys] if isinstance(keys, str) else keys
    fields = [fields] if isinstance(fields, str) else fields
    fields = fields or input_df.columns.to_list()
    input_df = input_df.reset_index()
    response_dict = nested_dict_to_flat(input_dict=asdict(input_obj, dict_factory=factory))
    response_df = pandas.DataFrame([response_dict])
    new_values = input_df[[*keys, *fields]].merge(response_df[keys], on=keys)
    if not new_values.empty:
        return new_values.to_dict('records')[0]
    return None


def set_obj_attr(input_obj: object, attr_set: dict):
    """
    For setting attributes of object

    :param input_obj: input object
    :param attr_set: dictionary with attribute: value pairs
    :return:  updated object
    """
    for key, value in attr_set.items():
        if hasattr(input_obj, key):
            setattr(input_obj, key, value)
    return input_obj


def update_calc_obj_version_from_df(input_df: pandas.DataFrame, calc_obj: CalculationPoint):
    """
    Updates the version number of CalculationPoint based on the dataframe (to reduce the queries to elastic update
    version number of the previous results)

    :param input_df: previous realised results
    :param calc_obj: CalculationPoint instance
    :return: updated instance
    """
    keys = [SENDER_MRID_KEY]
    fields = ['version_number']
    update_dict = None
    if not input_df.empty:
        update_dict = get_obj_fields_from_dataframe(input_df=input_df, input_obj=calc_obj, keys=keys, fields=fields)
    if update_dict is not None:
        update_dict = {k: str(int(v) + 1) for k, v in update_dict.items() if k in fields}
        calc_obj = set_obj_attr(input_obj=calc_obj, attr_set=update_dict)
    return calc_obj


def get_unified_value(input_df: pandas.DataFrame,
                      column_name: str,
                      strategy: ExceededPercentType = PY_EXCEEDED_PERCENT):
    """
    Gets value from dataframe column based on strategy

    :param input_df: input dataframe
    :param column_name: column for the value
    :param strategy: criteria for the value: min, max or median (default is max)
    :return: found value
    """
    if strategy == ExceededPercentType.MIN:
        output = input_df[column_name].min()
    elif strategy == ExceededPercentType.MEDIAN:
        output = input_df[column_name].median()
    else:
        output = input_df[column_name].max()
    return output


def divide_arrays(numerator, denominator, default_value = 2):
    """
    Divides arrays, returns default value if numerator is not zero and denominator is, if both are zero then return zero

    :param numerator: values to be divided
    :param denominator: values that divide
    :param default_value: return this value if numerator is not zero and denominator is
    :return: numpy array
    """
    numerator = numerator.to_numpy()
    denominator = denominator.to_numpy()
    result = numpy.empty_like(denominator, dtype=float)
    mask_valid = denominator != 0
    result[mask_valid] = numerator[mask_valid] / denominator[mask_valid]
    mask_zero_by_zero = (denominator == 0) & (numerator == 0)
    result[mask_zero_by_zero] = 0
    mask_div_by_zero = (denominator == 0) & (numerator != 0)
    result[mask_div_by_zero] = default_value
    return result


def calculate_exceeded_by_proportions(in_df: pandas.DataFrame,
                                      correct_initial: bool = False,
                                      report_exceeded: bool = False,
                                      sign_v: int | float = -1,
                                      strategy_percent: ExceededPercentType = PY_EXCEEDED_PERCENT,
                                      **kwargs):
    """
    First attempt to calculate excessive values

    :param in_df: input dataframe
    :param correct_initial: if true report new value
    :param report_exceeded: if true report difference
    :param sign_v: determine sign for the output values
    :param strategy_percent: how to choose percent
    :return: dataframe of exceeded liens
    """
    p_value = get_unified_value(input_df=in_df, column_name=PERCENTAGE_VALUE_KEY, strategy=strategy_percent)
    try:
        v_value = (in_df[in_df[PERCENTAGE_VALUE_KEY] == p_value][AVAILABLE_VALUE_KEY + PROP_SUFFIX].unique().item())
    except ValueError:
        slice_df = in_df[in_df[PERCENTAGE_VALUE_KEY] == p_value]
        in_df = slice_df[slice_df['version_number'] == slice_df['version_number'].max()].copy()
        v_value = in_df[AVAILABLE_VALUE_KEY + PROP_SUFFIX].unique().tolist()
        if len(v_value) > 1:
            logger.warning(f"Multiple versions detected")
        if len(v_value) > 0:
            v_value = v_value[0]
    in_df[NORM_FACTOR]= divide_arrays(numerator=in_df[AVAILABLE_VALUE_KEY + REAL_SUFFIX],
                                      denominator=in_df[AVAILABLE_VALUE_KEY + PROP_SUFFIX])
    sum_norm = in_df[NORM_FACTOR].sum()
    some_value = in_df['result_type'].unique().item()
    if sum_norm > 1:
        logger.error(f"For {some_value} sum of values exceeded. {v_value} was proposed at "
                     f"{p_value}%, sum of realised values at same percentage is "
                     f"{(in_df[NORM_FACTOR] * v_value).sum()}. Recalculating values...")
        in_df.loc[:, NORM_FACTOR] = in_df[NORM_FACTOR] / sum_norm
        if correct_initial:
            in_df.loc[:, EXCEEDED_KEY] = in_df[AVAILABLE_VALUE_KEY + PROP_SUFFIX] * in_df[NORM_FACTOR]
        else:
            in_df.loc[:, PERCENTAGE_VALUE_KEY] = p_value
            in_df.loc[:, EXCEEDED_KEY] = v_value * in_df[NORM_FACTOR]

        if report_exceeded:
            in_df.loc[:, AVAILABLE_VALUE_KEY] = (in_df[AVAILABLE_VALUE_KEY + REAL_SUFFIX] - in_df[EXCEEDED_KEY])
            in_df.loc[:, AVAILABLE_VALUE_KEY] = in_df[AVAILABLE_VALUE_KEY] * sign_v
        else:
            in_df.loc[:, AVAILABLE_VALUE_KEY] = in_df[EXCEEDED_KEY]
        in_df.loc[:, TIMESTAMP_KEY] = datetime.now()
        return in_df
    return pandas.DataFrame()


def calculate_exceeded_cascade(in_df: pandas.DataFrame,
                               report_exceeded: bool = False,
                               sign_v: int | float = -1,
                               **kwargs):
    """
    First attempt to calculate excessive values

    :param in_df: input dataframe
    :param report_exceeded: if true report difference
    :param sign_v: determine sign for the output values
    :return: dataframe of exceeded liens
    """
    some_value = in_df['result_type'].unique().item()
    running_sum = 0
    in_df['sum'] = in_df[AVAILABLE_VALUE_KEY + REAL_SUFFIX].groupby(in_df[PERCENTAGE_VALUE_KEY]).transform('sum')
    in_df['minus'] = in_df['sum'] - in_df[AVAILABLE_VALUE_KEY + PROP_SUFFIX]
    exceeded_cols = in_df[in_df['minus'] > 0]
    processed_groups = []
    grouped = in_df.groupby(in_df[PERCENTAGE_VALUE_KEY])
    for group_name in sorted(grouped.groups.keys(), reverse=True):
        group_df = grouped.get_group(group_name).copy()
        value_at_level = group_df[AVAILABLE_VALUE_KEY + PROP_SUFFIX].min() - running_sum
        used = group_df[AVAILABLE_VALUE_KEY + REAL_SUFFIX].sum()
        final_value = max(min(value_at_level, used), 0)
        group_df[EXCEEDED_KEY] = divide_arrays(numerator=group_df[AVAILABLE_VALUE_KEY + REAL_SUFFIX],
                                               denominator=group_df['sum'])
        group_df[EXCEEDED_KEY] = group_df[EXCEEDED_KEY] * final_value
        running_sum += final_value
        processed_groups.append(group_df)
    result = pandas.concat(processed_groups)
    v_value = exceeded_cols[AVAILABLE_VALUE_KEY + PROP_SUFFIX].sum()
    z_value = exceeded_cols[AVAILABLE_VALUE_KEY + REAL_SUFFIX].sum()
    p_value = ''.join([str(x) for x in exceeded_cols[PERCENTAGE_VALUE_KEY].unique().tolist()])
    if z_value > v_value:
        logger.error(f"For {some_value} sum of values exceeded. {v_value} was proposed at "
                     f"{p_value}%, sum of realised values at same percentage is "
                     f"{z_value}. Recalculating values...")
        if report_exceeded:
            result.loc[:, AVAILABLE_VALUE_KEY] = (result[AVAILABLE_VALUE_KEY + REAL_SUFFIX] - result[EXCEEDED_KEY])
            result.loc[:, AVAILABLE_VALUE_KEY] = result[AVAILABLE_VALUE_KEY] * sign_v
        else:
            result.loc[:, AVAILABLE_VALUE_KEY] = result[EXCEEDED_KEY]
        result.loc[:, TIMESTAMP_KEY] = datetime.now()
        final_df = result[(result[AVAILABLE_VALUE_KEY] - result[AVAILABLE_VALUE_KEY + REAL_SUFFIX]).abs() > 0]
        return final_df
    return pandas.DataFrame()



EXCEEDED_OPERATORS = {'PROPORTION': Operator(method=calculate_exceeded_by_proportions),
                      'CASCADE': Operator(method=calculate_exceeded_cascade)}


def compare_realised_values_to_proposed_values(realised_data: list = None,
                                               exceeded_check: ProcurementCalculationType = PY_EXCEEDED_TYPE,
                                               report_exceeded: bool = PY_EXCEEDED_REPORT,
                                               strategy_percent: ExceededPercentType = PY_EXCEEDED_PERCENT,
                                               correct_initial: bool = PY_EXCEEDED_INITIAL,
                                               operator_dict: dict = None,
                                               operator: ExceededEnumOperator=PY_EXCEEDED_OPERATOR,
                                               key_words: list = None):
    """
    For the realised results asks existing realised results for same parameters. Updates them and compares them with
    proposed values at the same parameters.

    :param operator: choose function
    :param operator_dict: operator selection
    :param exceeded_check: For bypassing
    :param realised_data: list of parsed results
    :param report_exceeded: True, calculates difference, False, returns new, proportional value
    :param strategy_percent: Strategy to choose common percentage (min, max or median of existing)
    :param correct_initial: whether to correct the existing one (True) or the common one (False
    :param key_words: use this for filtering to latest value
    :return: Dataframe of exceeded CalculationPoints
    """
    operator_dict = operator_dict or EXCEEDED_OPERATORS
    operator_func = operator_dict.get(operator.value)
    exceeded_lines = []
    key_words = key_words or UNIQUE_COLUMN_KEYWORDS
    sign_v = int(PY_EXCEEDED_SIGN) / abs(int(PY_EXCEEDED_SIGN))
    for result in realised_data:
        if not (result.calculation_type == exceeded_check or exceeded_check == ProcurementCalculationType.ALL):
            continue
        for single_list in result.object_list:
            for calc_ob in single_list:
                proposed, realised = get_proposed_realised_for_calc_ob(calc_ob=calc_ob, key_words=key_words)
                if not proposed.empty:
                    calc_ob = update_calc_obj_version_from_df(input_df=realised, calc_obj=calc_ob)
                    realised = update_dataframe_from_calc_object(input_df=realised,
                                                                 calc_obj=calc_ob,
                                                                 keys=SENDER_MRID_KEY)
                    realised_df = (realised[[PERCENTAGE_VALUE_KEY, AVAILABLE_VALUE_KEY, SENDER_MRID_KEY]]
                                   .rename(columns={SENDER_MRID_KEY: RECEIVER_MRID_KEY}))
                    exceeded = proposed.merge(realised_df, on=PERCENTAGE_VALUE_KEY, suffixes=(PROP_SUFFIX, REAL_SUFFIX))
                    if operator_func is not None:
                        triggered_df = operator_func.method(in_df=exceeded,
                                                            correct_initial= correct_initial,
                                                            report_exceeded=report_exceeded,
                                                            sign_v=sign_v,
                                                            strategy_percent=strategy_percent)
                        if not triggered_df.empty:
                            exceeded_lines.append(triggered_df)
    return pandas.concat(exceeded_lines) if len(exceeded_lines) >= 1 else pandas.DataFrame()


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    file_name = r"E:\margus.ratsep\sizing_of_reserves\reports\Realised-NCPB-ATC-02-08-2025_14-22-51.xlsx"
    environment_task = get_task_from_environment()
    all_results = parse_excel_to_calculation_results(path_to_excel=file_name,
                                                     task=environment_task,
                                                     included_sheets=EXCEL_INCLUDED_SHEETS,
                                                     excluded_sheets=EXCEL_EXCLUDED_SHEETS)
    final_result = compare_realised_values_to_proposed_values(realised_data=all_results)

    print("Done")
