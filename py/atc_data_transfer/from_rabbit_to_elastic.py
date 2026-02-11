import logging
import uuid

from py.common.to_elastic_logger import initialize_custom_logger
from py.data_classes.enums import MessageACK
from py.handlers.elastic_handler import ElkHandler as Elastic, PY_ELASTICSEARCH_HOST, PY_PROCUREMENT_ATC_INDEX, \
    PY_PROCUREMENT_REALISED_INDEX
from py.handlers.rabbit_handler import BodyHandlerInterface, ProcurementBlockingClient, \
    ProcurementRMQConsumer, PY_RMQ_LISTEN_CONTINUOUSLY, string_for_edx, flatten_tuple, get_bytes_from_args

import config
from py.common.config_parser import parse_app_properties
from py.common.functions import convert_input
from py.parsers.file_types import is_json, is_xml, is_xlsx, handle_json, handle_xlsx, parse_input_by_type
from py.procurement.transparency_data import xml_to_elastic
from py.parsers.xlsx_to_calculation_result import parse_excel_to_calculation_results
from py.parsers.xml_to_calculation_result import (PY_PROD_ATC_INDEX_KEYS, generate_ids_from_dict,
                                                  parse_xml_dataframe_to_calculation_result)
from py.procurement.constants import PY_PROPOSED_RMQ_HEADERS, PY_REALISING_RMQ_HEADERS, PY_EXCEEDED_RMQ_HEADERS, RCC, \
    AVAILABLE_SENDERS
from py.procurement.dummy_tso import TestTso
from py.procurement.realised_data_check import compare_realised_values_to_proposed_values, RealisedDataCollector
from py.procurement.procurement_output import results_to_elastic

# Disabling triplets library logging at INFO level
logger = logging.getLogger(__name__)

parse_app_properties(globals(), config.paths.config.atc_input)

PY_ATC_HEADERS = convert_input(ATC_HEADERS)


def check_realised_values(realised_data: list = None):
    """
    Entry point for realised data check

    :param realised_data: input from tsos
    :return: output from realised check
    """
    final_result = compare_realised_values_to_proposed_values(realised_data=realised_data)
    return final_result


class ATCInputHandler(BodyHandlerInterface):

    """
    Handler for handling messages containing ATC data.

    :param filter_dict: filter messages
    :param elastic_address: specify address of elastic server
    :param elastic_index: specify index in elastic
    """

    def __init__(self,
                 filter_dict: dict = None,
                 elastic_address: str = PY_ELASTICSEARCH_HOST,
                 elastic_index: str = PY_PROCUREMENT_ATC_INDEX):
        """
        Constructor
        """

        super().__init__(filter_dict)
        self.elastic_address = elastic_address
        self.elastic_index = elastic_index

    def handle(self, *args, **kwargs):
        """
        Handles input message.

        :param args: unnamed arguments
        :param kwargs: named arguments
        :return: unnamed arguments
        """
        args = flatten_tuple(args)
        headers = self.filter_message(**kwargs)
        if headers is None:
            return args
        if (payload := get_bytes_from_args(*args)) is None:
            return args
        parsed_list = parse_input_by_type(input_data=payload,
                                          type_dict={is_xml: xml_to_elastic,
                                                     is_json: handle_json,
                                                     is_xlsx: handle_xlsx},
                                          caller=self)
        if parsed_list:
            logger.info(f"Found {len(parsed_list)} entries of ATC, sending to elastic")
            dicts_with_ids = generate_ids_from_dict(input_data=parsed_list, id_fields=PY_PROD_ATC_INDEX_KEYS)
            Elastic.send_to_elastic_bulk(json_message_list=dicts_with_ids,
                                         server=self.elastic_address,
                                         index=self.elastic_index)
        return args, MessageACK.ACK


def get_mrid_from_from_message_id(input_str, senders: list = None, attr_name: str = 'mRID'):
    """
    Workaround for testing. As Excel does not contain any metadata and headers are truncated when coming from
    edx then for distinguishing different "dummy" tsos (same edx address) then mrid from file name is used

    :param input_str: input string
    :param senders: list of senders
    :param attr_name: keyword to search
    :return: sender as a match or None
    """
    senders = senders or AVAILABLE_SENDERS
    if senders is None:
        logger.info(f"No senders loaded, returning")
        return None
    try:
        match = next(iter([x for x in senders if hasattr(x, attr_name) and getattr(x, attr_name) in input_str]), None)
        if match is None:
            match = next(iter([x for x in senders
                               if hasattr(x, attr_name) and string_for_edx(getattr(x, attr_name)) in input_str]), None)
        return match
    except TypeError:
        return None


class ProposedValuesHandler(BodyHandlerInterface):
    """
    Handler for handling messages containing proposed data (For testing purposes only).

    :param filter_dict: filter messages
    :param send_to_dummy_tso: whether to initiate dummy tso
    """

    def __init__(self, filter_dict = None, send_to_dummy_tso:bool = True):
        """
        Constructor
        """
        super().__init__(filter_dict)
        self.dummy_tsos = []
        self.send_to_dummy_tso = send_to_dummy_tso

    def get_dummy_tso(self, sender: object = None):
        response = None
        if len(self.dummy_tsos) > 0:
            if sender is not None:
                response = next(iter(x for x in self.dummy_tsos if x.sender == sender), None)
        return response

    def add_dummy_tso(self, sender: object = None, receiver: object = None):
        response = self.get_dummy_tso(sender=sender)
        if response:
            logger.info(f"Dummy TSO mRID: {response.sender.mRID} already exists")
        if response is None:
            response = TestTso(sender_value=sender, receiver_value=receiver)
            logger.info(f"Created Dummy TSO mRID: {response.sender.mRID}")
            self.dummy_tsos.append(response)
        return response

    def handle(self, *args, **kwargs):
        """
        Handles input message.

        :param args: unnamed arguments
        :param kwargs: named arguments
        :return: unnamed arguments
        """
        args = flatten_tuple(args)
        headers = self.filter_message(**kwargs)
        if headers is None:
            return args
        headers.senderApplication = RCC
        if (payload := get_bytes_from_args(*args)) is None:
            return args
        # Comment this out when going alive. Real TSOs have unique addresses
        receiver = get_mrid_from_from_message_id(input_str=headers.baMessageID, attr_name='mRID')
        if receiver is not None:
            headers.receiverCode = receiver
        parsed_list = parse_input_by_type(input_data=payload,
                                          type_dict={is_xml: parse_xml_dataframe_to_calculation_result,
                                                     is_xlsx: parse_excel_to_calculation_results},
                                          caller=self)
        if parsed_list:
            if self.send_to_dummy_tso:
                logger.info(f"{self.__class__.__name__}: Found {len(parsed_list)} entries, to dummy TSO")
                entity = next(iter([getattr(x, 'receivers') for x in parsed_list if hasattr(x, 'receivers')]), None)
                entity = entity[0] if isinstance(entity, list) and len(entity) == 1 else None
                # Flip sender and receiver
                receiver = headers.senderApplication or  next(iter(getattr(x, 'sender')
                                                                   for x in parsed_list if hasattr(x, 'sender')), None)
                sender = headers.receiverCode  or entity
                dummy_tso = self.add_dummy_tso(sender=sender, receiver=receiver)
                dummy_tso.handle_input(input_data=parsed_list)
            else:
                logger.info(f"{self.__class__.__name__}: Found {len(parsed_list)} entries, bypassing")
        return args, MessageACK.ACK


class ExceededValuesHandler(ProposedValuesHandler):
    """
    For handling exceeded values
    """
    pass


class RealisedValuesHandler(BodyHandlerInterface):
    """
    Handler for handling messages containing realised data from TSOs.

    :param filter_dict: filter messages
    :param elastic_address: specify address of elastic server
    :param elastic_index: specify index in elastic
    """

    def __init__(self,
                 filter_dict = None,
                 to_elastic: bool = True,
                 elastic_address: str = PY_ELASTICSEARCH_HOST,
                 elastic_index: str = PY_PROCUREMENT_REALISED_INDEX):
        """
        Constructor
        """
        super().__init__(filter_dict)
        self.to_elastic = to_elastic
        self.realised_gatherer = RealisedDataCollector()
        self.elastic_address = elastic_address
        self.elastic_index = elastic_index
        self.realised_objects = []

    def send_out_exceeded_values(self):
        self.realised_gatherer.parse_input()

    def handle(self, *args, **kwargs):
        """
        Handles input message.

        :param args: unnamed arguments
        :param kwargs: named arguments
        :return: unnamed arguments
        """
        args = flatten_tuple(args)
        headers = self.filter_message(**kwargs)
        if headers is None:
            return args
        if (payload := get_bytes_from_args(*args)) is None:
            return args
        parsed_list = parse_input_by_type(input_data=payload,
                                          type_dict={is_xml: parse_xml_dataframe_to_calculation_result,
                                                     is_xlsx: parse_excel_to_calculation_results},
                                          caller=self)
        if parsed_list:
            logger.info(f"Found {len(parsed_list)} entries of realised values, sending to elastic")
            #exceed_values = check_realised_values(realised_data=parsed_list)
            #if not exceed_values.empty:
            #    self.realised_gatherer.add_to_buffer(exceed_values)
            if self.to_elastic:
                for item in parsed_list:
                    results_to_elastic(result=item, elastic_index=PY_PROCUREMENT_REALISED_INDEX)
        return args, MessageACK.ACK


if __name__ == '__main__':

    initialize_custom_logger(extra_fields={"Job": "Rabbit listener", "Job_id": str(uuid.uuid4())})
    handler_set = [
        #ATCInputHandler(filter_dict=PY_ATC_HEADERS), # Old, legacy ATC retrieving process
        #ProposedValuesHandler(filter_dict=PY_PROPOSED_RMQ_HEADERS), # Dummy TSOs
        RealisedValuesHandler(filter_dict=PY_REALISING_RMQ_HEADERS, to_elastic=True), # Check realised, send exceeded
        ExceededValuesHandler(filter_dict=PY_EXCEEDED_RMQ_HEADERS, send_to_dummy_tso=False) # Consume exceeded
                   ]

    if not PY_RMQ_LISTEN_CONTINUOUSLY:
        # Blocking client
        client = ProcurementBlockingClient(message_handlers=handler_set)
        client.consume_until_empty()
        re_handler = next(iter(x for x in handler_set if isinstance(x, RealisedValuesHandler)), None)
        if re_handler is not None:
            re_handler.send_out_exceeded_values()

        print("Done")
    else:
        # RabbitMQ consumer implementation
        consumer = ProcurementRMQConsumer(message_handlers=handler_set)
        try:
            consumer.run()
        except KeyboardInterrupt:
            consumer.stop()