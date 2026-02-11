import copy
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from py.data_classes.elastic.elastic_data_models import CalculationPoint
from py.data_classes.enums import ProcessType, ProcurementCalculationType
from py.data_classes.results.calculation_result_atc import CalculationResultATC
from py.data_classes.results.calculation_result_bid import CalculationResultBid
from py.data_classes.results.calculation_result_main import CalculationResult
from py.data_classes.task_classes import ProcurementCalculationTask, MessageCommunicator
from py.handlers.rabbit_handler import PY_RMQ_EXCHANGE
from py.parsers.xml_to_calculation_result import filter_list
from py.procurement.constants import PY_OUTPUT_SEND_XML_TO_MINIO, \
    PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE, PY_OUTPUT_SEND_XML_TO_RABBIT, PY_REALISING_RMQ_HEADERS, XML_FOLDER_TO_STORE, \
    PY_OUTPUT_SEND_XLSX_TO_MINIO, PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE, EXCEL_FOLDER_TO_STORE, \
    PY_OUTPUT_SEND_XLSX_TO_RABBIT
from py.handlers.elastic_handler import PY_PROCUREMENT_REALISED_INDEX
from py.procurement.procurement_common import get_task_from_environment
from py.procurement.procurement_output import handle_parsed_output

logger = logging.getLogger(__name__)

NCPB_INCLUDED_KEYWORDS = ['domain', 'bid', 'direction']
NCPB_EXCLUDED_KEYWORDS = ['in', 'out']
ATC_INCLUDED_KEYWORDS = ['in_domain', 'out_domain', 'ATC', 'capacity']
ATC_EXCLUDED_KEYWORDS = []

# CUSTOM_PERCENTAGE_VALUE = 90
CUSTOM_PERCENTAGE_VALUE = None


def check_attribute(input_object: object, attr_name: str = None, attr_value: Any = None):
    """
    Checks if attribute with value exists in object

    :param input_object: object to check
    :param attr_name: attribute name
    :param attr_value: attribute value
    :return: True if exists, false otherwise
    """
    return getattr(input_object, attr_name) == attr_value if (attr_name and hasattr(input_object, attr_name)) else False


def check_attributes(input_object: object, attribute_pairs: dict):
    """
    Checks if attributes with values (dict) exist in given object

    :param input_object: object to check
    :param attribute_pairs: dictionary where keys are attributes, values are attribute values
    :return: True if everything exist, False otherwise
    """
    output = True
    for attr_name, attr_value in attribute_pairs.items():
        output = check_attribute(input_object=input_object, attr_name=attr_name, attr_value=attr_value)
        if not output:
            return output
    return output


def reorganize_by_quantiles(results: list[CalculationResult]):
    """
    Squeezes quantiles from multiple results to one (if applicable). Note that this is used for dummy tso

    :param results: list of CalculationResult instances
    :return: new CalculationResult instance
    """
    initial_calc_result = next(iter(x for x in results))
    all_quantiles = []
    for x in results:
        for quantile in x.quantiles:
            if len(all_quantiles) == 0 or not any(iter(x for x in all_quantiles if x.same_time_slice(quantile))):
                all_quantiles.append(quantile)
    initial_calc_result.update_object_list_from_quantiles(quantiles=all_quantiles)
    return [initial_calc_result]

@dataclass
class TypeFilter:

    enum_value: Enum
    included_keywords: list = None
    excluded_keywords: list = None

    def filter(self, input_list: list):
        """
        Checks if list of strings contains included keywords and does not include excluded keywords. If these strings
        exist returns set enum

        :param input_list: list of strings
        :return: enum if filtered list has values, None otherwise
        """
        responses = filter_list(input_data=input_list, included=self.included_keywords, excluded=self.excluded_keywords)
        if responses and len(responses) > 0:
            return self.enum_value
        return None

ATC_TYPE_FILTER = TypeFilter(enum_value=ProcurementCalculationType.ATC,
                             included_keywords=ATC_INCLUDED_KEYWORDS,
                             excluded_keywords=ATC_EXCLUDED_KEYWORDS)

NCPB_TYPE_FILTER = TypeFilter(enum_value=ProcurementCalculationType.NCPB,
                              included_keywords=NCPB_INCLUDED_KEYWORDS,
                              excluded_keywords=NCPB_EXCLUDED_KEYWORDS)


def generate_dummy_results(input_result: CalculationResult, single_task: ProcurementCalculationTask):
    new_result = copy.deepcopy(input_result)
    new_result.init_from_task(single_task)
    new_result.process_type = ProcessType.REALISED
    input_values = new_result.object_list
    output_values = []
    for input_value in input_values:
        products = {}
        single_output = []
        for point in input_value:
            if isinstance(point, CalculationPoint):
                id_value = point.generate_product_id()
                if not id_value in products.keys():
                    products[id_value] = []
                products[id_value].append(point)
        for product_key, product_value in products.items():
            output_value = random.choice(product_value)
            if float(output_value.available.value) > 0:
                output_value.available.value = random.uniform(0, 2 * output_value.available.value)
                # output_value.available.value = random.uniform(0, output_value.available.value)
            if CUSTOM_PERCENTAGE_VALUE is not None and float(CUSTOM_PERCENTAGE_VALUE) >= 90.0:
                output_value.percentage_level.value = CUSTOM_PERCENTAGE_VALUE
            # if float(output_value.available.value) <= 0:
            #     output_value.percentage_level.value = None
            single_output.append(output_value)
        output_values.append(single_output)
    new_result.elastic_index = PY_PROCUREMENT_REALISED_INDEX
    new_result.set_object_list(output_values)
    new_result.update_quantiles_from_object_list()
    return new_result


def generate_dummy_responses(input_results,
                             receiver: MessageCommunicator | list = None,
                             sender: MessageCommunicator | list = None,
                             task: ProcurementCalculationTask = None):
    """
    For creating dummy responses

    :param task:
    :param receiver:
    :param sender:
    :param input_results: calculation results (the ones meant to send to minio
    :return: responses
    """
    task = task or get_task_from_environment()
    output_results = []
    new_tasks = []
    receiver = receiver or task.receivers
    sender = sender or task.sender
    receiver = [receiver] if not isinstance(receiver, list) else receiver
    sender = [sender] if not isinstance(sender, list) else sender
    for single_receiver in receiver:
        new_task = copy.deepcopy(task)
        new_task.receivers = sender
        new_task.sender = single_receiver
        new_tasks.append(new_task)
    for input_result in input_results:
        for single_task in new_tasks:
            output_results.append(generate_dummy_results(input_result=input_result, single_task=single_task))
    return output_results


class TestTso:

    def __init__(self, receiver_value = None, sender_value = None):
        self.receiver = receiver_value
        self.sender = sender_value
        self.atc_values = None
        self.ncpb_values = None
        self.realised_values = None
        self.task = get_task_from_environment()
        self.prefixes = ['Realised']

    def get_me(self):
        if self.sender is not None:
            return self.sender.to_dict()
        return None

    def generate_realised_values(self, results: list):
        """
        Generates and sends out the results

        :param results: list of results
        :return:
        """
        self.realised_values = generate_dummy_responses(input_results=results,
                                                        receiver=self.sender,
                                                        sender=self.receiver,
                                                        task=self.task)
        self.realised_values = list({x.calculation_type: x for x in self.realised_values}.values())
        for single_value in self.realised_values:
            single_value.update_sender()
        # str_value = {x.__class__.__name__: len(x.input_data.index) for x in self.realised_values}
        logger.info(f"TSO mRID: {self.sender.mRID}: Generated response values (entries: {len(self.realised_values)})")
        handle_parsed_output(results=self.realised_values,
                             output_to_elastic=False,
                             elastic_index=None,
                             prefixes=self.prefixes,
                             output_xml_minio=PY_OUTPUT_SEND_XML_TO_MINIO,
                             output_xml_rabbit=PY_OUTPUT_SEND_XML_TO_RABBIT,
                             output_xml_local=PY_OUTPUT_SAVE_XML_TO_LOCAL_STORAGE,
                             rabbit_exchange=PY_RMQ_EXCHANGE,
                             rabbit_headers=PY_REALISING_RMQ_HEADERS,
                             xml_local_path=XML_FOLDER_TO_STORE,
                             output_xlsx_minio=PY_OUTPUT_SEND_XLSX_TO_MINIO,
                             output_xlsx_rabbit=PY_OUTPUT_SEND_XLSX_TO_RABBIT,
                             output_xlsx_local=PY_OUTPUT_SAVE_XLSX_TO_LOCAL_STORAGE,
                             add_corrected_ncpb=False,
                             xlsx_local_path=EXCEL_FOLDER_TO_STORE)
        logger.info(f"TSO mRID: {self.sender.mRID}  finished")

    def send_realised_values(self):
        """
        Generates realised values if both are present

        :return:
        """
        if self.atc_values is not None and self.ncpb_values is not None:
            self.atc_values = reorganize_by_quantiles(self.atc_values)
            self.ncpb_values = [x for x in self.ncpb_values
                                if not check_attributes(x, attribute_pairs={'NCPB_updated': True})]
            self.ncpb_values = reorganize_by_quantiles(self.ncpb_values)
            if self.realised_values is None:
                self.generate_realised_values(self.atc_values + self.ncpb_values)

    def handle_input(self, input_data):
        """
        Main function to generate responses

        :param input_data: (list of) CalculationResult
        :return: None
        """
        if not isinstance(input_data, list):
            input_data = [input_data]
        for input_datum in input_data:
            if isinstance(input_datum, CalculationResultATC):
                self.atc_values = self.atc_values or []
                self.atc_values.append(input_datum)
                logger.info(f"TSO mRID: : "
                            f"Received Proposed ATC values, in total {len(self.atc_values)} entries")
            if isinstance(input_datum, CalculationResultBid):
                self.ncpb_values = self.ncpb_values or []
                self.ncpb_values.append(input_datum)
                logger.info(f"TSO mRID: : "
                            f"Received Proposed NCPB values, in total {len(self.ncpb_values)} entries")
        self.send_realised_values()
