import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, asdict
from io import BytesIO
from random import choice
from string import ascii_uppercase

import config
import logging
import time

from typing import List, Any
from py.common.config_parser import parse_app_properties
from py.common.functions import parse_to_type, check_dict_to_dataclass, filter_dict_by_enum
from py.data_classes.enums import OutputFileType, ValueOfEnum, MessageACK
from py.data_classes.task_classes import MessageCommunicator
from py.procurement.constants import AVAILABLE_SENDERS

logger = logging.getLogger(__name__)

try:
    from brcc_apis.rabbit import BlockingClient, RMQConsumer
except ModuleNotFoundError:

    import functools
    import pika

    from concurrent.futures import ThreadPoolExecutor


    class BlockingClient:

        def __init__(self,
                     host: str,
                     port: int,
                     username: str,
                     password: str,
                     message_converter: object | None = None,
                     message_handler: object | None = None,
                     ):
            self.connection_params = {
                'host': host,
                'port': port,
                'credentials': pika.PlainCredentials(username, password)
            }
            self.message_converter = message_converter
            self.message_handler = message_handler
            self._connect()
            self.consuming = False

        def _connect(self):
            # Connect to RabbitMQ server
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(**self.connection_params)
            )
            self.publish_channel = self.connection.channel()
            self.consume_channel = self.connection.channel()

        def publish(self, payload: str, exchange_name: str, headers: dict | None = None, routing_key: str = ''):
            # Publish message
            self.publish_channel.basic_publish(
                exchange=exchange_name,
                routing_key=routing_key,
                body=payload,
                properties=pika.BasicProperties(
                    headers=headers
                )
            )

        def get_single_message(self, queue: str, auto_ack: bool = True):
            """
            Attempt to fetch a single message from the specified queue.

            :param queue: The name of the queue to fetch the message from.
            :param auto_ack: Whether to automatically acknowledge the message. Defaults to True.
            :return: The method frame, properties, and body of the message if available; otherwise, None.
            """

            # Stop previous consume
            if self.consuming:
                self.consume_stop()

            method_frame, properties, body = self.consume_channel.basic_get(queue, auto_ack=auto_ack)

            if method_frame:
                logger.info(f"Received message from {queue}: {properties}")

                # Convert message
                if self.message_converter:
                    try:
                        body, content_type = self.message_converter.convert(body)
                        properties.content_type = content_type
                        logger.info(f"Message converted")
                    except Exception as error:
                        logger.error(f"Message conversion failed: {error}")
                return method_frame, properties, body
            else:
                logger.info(f"No message available in queue {queue}")
                return None, None, None

        def consume_start(self, queue: str, callback: object | None = None, auto_ack: bool = True):

            # Stop previous consume
            if self.consuming:
                self.consume_stop()

            # Set up consumer
            if not callback:
                callback = lambda ch, method, properties, body: logger.info(
                    f"Received message: {properties} (No callback processing)")

            self.consume_channel.basic_consume(
                queue=queue,
                on_message_callback=callback,
                auto_ack=auto_ack
            )

            logger.info(f"Waiting for messages in {queue}. To exit press CTRL+C")

            try:
                self.consume_channel.start_consuming()
                self.consuming = True
            except KeyboardInterrupt:
                self.consume_stop()

        def consume_stop(self):
            self.consume_channel.stop_consuming()
            self.consuming = False

        def close(self):

            # Stop consuming
            if self.consuming:
                self.consume_stop()

            # Close the connection
            if self.connection.is_open:
                self.connection.close()

        def __del__(self):
            # Destructor to ensure the connection is closed properly
            self.close()


    class RMQConsumer:
        """This is an example consumer that will handle unexpected interactions
        with RabbitMQ such as channel and connection closures.

        If RabbitMQ closes the connection, it will reopen it. You should
        look at the output, as there are limited reasons why the connection may
        be closed, which usually are tied to permission related issues or
        socket timeouts.

        If the channel is closed, it will indicate a problem with one of the
        commands that were issued and that should surface in the output as well.
        """

        def __init__(self,
                     host: str,
                     port: int,
                     vhost: str,
                     username: str,
                     password: str,
                     que: str | None = None,
                     heartbeat: str | int | None = None,
                     message_handlers: List[object] | None = None,
                     message_converter: object | None = None,
                     ):
            """Create a new instance of the consumer class, passing in the AMQP
            URL used to connect to RabbitMQ.
            """
            self.message_handlers = message_handlers
            self.message_converter = message_converter

            self._connection = None
            self._channel = None
            self._closing = False
            self._consumer_tag = None
            self._consuming = False
            # In production, experiment with higher prefetch values
            # for higher consumer throughput
            self._prefetch_count = 1

            self._host = host
            self._port = port
            self._vhost = vhost
            self._que = que
            self._username = username

            self._executor = ThreadPoolExecutor()
            self._executor_stopped = False

            self._connection_parameters = pika.ConnectionParameters(host=self._host,
                                                                    port=self._port,
                                                                    virtual_host=self._vhost,
                                                                    credentials=pika.PlainCredentials(username,
                                                                                                      password))
            self.set_heartbeat(heartbeat=heartbeat)

        def set_heartbeat(self, heartbeat: str | int):
            """
            Brings heartbeat parameter out to be configured
            NB! guard is to added not to switch the heartbeat off
            :param heartbeat: new heartbeat value to send to server
            """
            if heartbeat:
                if isinstance(heartbeat, str):
                    try:
                        heartbeat = int(heartbeat)
                    except ValueError:
                        heartbeat = None
                # Do not switch the heartbeat off
                if heartbeat and heartbeat > 0:
                    self._connection_parameters.heartbeat = heartbeat

        def connect(self):
            """This method connects to RabbitMQ, returning the connection handle.
            When the connection is established, the on_connection_open method
            will be invoked by pika.

            :rtype: pika.SelectConnection

            """
            logger.info(f"Connecting to {self._host}:{self._port} @ {self._vhost} as {self._username}")

            return pika.SelectConnection(
                parameters=self._connection_parameters,
                on_open_callback=self.on_connection_open,
                on_open_error_callback=self.on_connection_open_error,
                on_close_callback=self.on_connection_closed)

        def close_connection(self):
            self._consuming = False
            if self._connection.is_closing or self._connection.is_closed:
                logger.info("Connection is closing or already closed")
            else:
                logger.info("Closing connection")
                self._connection.close()

        def on_connection_open(self, _unused_connection):
            """This method is called by pika once the connection to RabbitMQ has
            been established. It passes the handle to the connection object in
            case we need it, but in this case, we'll just mark it unused.
            :param pika.SelectConnection _unused_connection: The connection
            """
            logger.info("Connection opened")
            self.open_channel()

        def on_connection_open_error(self, _unused_connection, err):
            """This method is called by pika if the connection to RabbitMQ
            can't be established.
            :param pika.SelectConnection _unused_connection: The connection
            :param Exception err: The error
            """
            logger.error(f"Connection open failed", exc_info=err)
            self.reconnect()

        def on_connection_closed(self, _unused_connection, reason):
            """This method is invoked by pika when the connection to RabbitMQ is
            closed unexpectedly. Since it is unexpected, we will reconnect to
            RabbitMQ if it disconnects.
            :param _unused_connection: The closed connection obj
            :param Exception reason: exception representing reason for loss of connection.
            """
            self._channel = None
            if self._closing:
                self._connection.ioloop.stop()
            else:
                logger.warning(f"Connection closed, reconnect necessary: {reason}")
                self.reconnect()

        def reconnect(self):
            """Will be invoked if the connection can't be opened or is
            closed. Indicates that a reconnect is necessary then stops the
            ioloop.
            """
            self.stop()

        def open_channel(self):
            """Open a new channel with RabbitMQ by issuing the Channel.Open RPC
            command. When RabbitMQ responds that the channel is open, the
            on_channel_open callback will be invoked by pika.
            """
            logger.info("Creating a new channel")
            self._connection.channel(on_open_callback=self.on_channel_open)

        def on_channel_open(self, channel):
            """This method is invoked by pika when the channel has been opened.
            The channel object is passed in so we can make use of it.
            Since the channel is now open, we'll declare the exchange to use.
            :param pika.channel.Channel channel: The channel object
            """
            logger.info("Channel opened")
            self._channel = channel
            self.add_on_channel_close_callback()
            self.set_qos()

        def add_on_channel_close_callback(self):
            """This method tells pika to call the on_channel_closed method if
            RabbitMQ unexpectedly closes the channel.
            """
            logger.info("Adding channel close callback")
            self._channel.add_on_close_callback(self.on_channel_closed)

        def on_channel_closed(self, channel, reason):
            """Invoked by pika when RabbitMQ unexpectedly closes the channel.
            Channels are usually closed if you attempt to do something that
            violates the protocol, such as re-declare an exchange or queue with
            different parameters. In this case, we'll close the connection
            to shut down the object.
            :param channel: The closed channel
            :param Exception reason: why the channel was closed
            """
            logger.warning(f"Channel {channel} was closed: {reason}")
            self.close_connection()

        def set_qos(self):
            """This method sets up the consumer prefetch to only be delivered
            one message at a time. The consumer must acknowledge this message
            before RabbitMQ will deliver another one. You should experiment
            with different prefetch values to achieve desired performance.
            """
            self._channel.basic_qos(
                prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok)

        def on_basic_qos_ok(self, _unused_frame):
            """Invoked by pika when the Basic.QoS method has completed. At this
            point we will start consuming messages by calling start_consuming
            which will invoke the needed RPC commands to start the process.
            :param pika.frame.Method _unused_frame: The Basic.QosOk response frame
            """
            logger.info(f"QOS set to: {self._prefetch_count}")
            self.start_consuming()

        def start_consuming(self):
            """This method sets up the consumer by first calling
            add_on_cancel_callback so that the object is notified if RabbitMQ
            cancels the consumer. It then issues the Basic.Consume RPC command
            which returns the consumer tag that is used to uniquely identify the
            consumer with RabbitMQ. We keep the value to use it when we want to
            cancel consuming. The on_message method is passed in as a callback pika
            will invoke when a message is fully received.
            """
            logger.info("Issuing consumer related RPC commands")
            self.add_on_cancel_callback()
            self._consumer_tag = self._channel.basic_consume(self._que, self.on_message)
            self._consuming = True

        def add_on_cancel_callback(self):
            """Add a callback that will be invoked if RabbitMQ cancels the consumer
            for some reason. If RabbitMQ does cancel the consumer,
            on_consumer_cancelled will be invoked by pika.
            """
            logger.info("Adding consumer cancellation callback")
            self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

        def on_consumer_cancelled(self, method_frame):
            """Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer
            receiving messages.
            :param pika.frame.Method method_frame: The Basic.Cancel frame
            """
            logger.info(f"Consumer was cancelled remotely, shutting down: {method_frame}")

            if self._channel:
                self._channel.close()

        def _process_messages(self, basic_deliver, properties, body):

            ack = True

            # Convert if needed
            if self.message_converter:
                try:
                    logger.info(f"Converting message with converter: {self.message_converter.__class__.__name__}")
                    body, content_type = self.message_converter.convert(body, properties=properties)
                    properties.content_type = content_type
                    logger.info(f"Message converted")
                except Exception as error:
                    logger.error(f"Message conversion failed: {error}", exc_info=True)
                    ack = False

            if self.message_handlers:
                for message_handler in self.message_handlers:
                    try:
                        logger.info(f"Handling message with handler: {message_handler.__class__.__name__}")
                        body = message_handler.handle(body, properties=properties)
                    except Exception as error:
                        logger.error(f"Message handling failed: {error}", exc_info=True)
                        ack = False
                        break

            if ack:
                self.acknowledge_message(basic_deliver.delivery_tag)

        def on_message(self, _unused_channel, basic_deliver, properties, body):
            """Invoked by pika when a message is delivered from RabbitMQ. The
            channel is passed for your convenience. The basic_deliver object that
            is passed in carries the exchange, routing key, delivery tag and
            a redelivered flag for the message. The properties passed in is an
            instance of BasicProperties with the message properties and the body
            is the message that was sent.

            :param pika.channel.Channel _unused_channel: The channel object
            :param basic_deliver: basic_deliver method
            :param properties: properties
            :param bytes body: The message body
            """
            logger.info(
                f"Received message # {basic_deliver.delivery_tag} from {properties.app_id} meta: {properties.headers}")
            logger.debug(f"Message body: {body}")
            self._executor.submit(self._process_messages, basic_deliver, properties, body)

        def acknowledge_message(self, delivery_tag):
            """Acknowledge the message delivery from RabbitMQ by sending a
            Basic.Ack RPC method for the delivery tag.

            :param int delivery_tag: The delivery tag from the Basic.Deliver frame
            """
            logger.info(f"Acknowledging message {delivery_tag}")
            self._channel.basic_ack(delivery_tag)

        def stop_consuming(self):
            """Tell RabbitMQ that you would like to stop consuming by sending the
            Basic.Cancel RPC command.
            """
            if self._channel:
                logger.info("Sending a Basic.Cancel RPC command to RabbitMQ")
                cb = functools.partial(self.on_cancelok, userdata=self._consumer_tag)
                self._channel.basic_cancel(self._consumer_tag, cb)

        def on_cancelok(self, _unused_frame, userdata):
            """This method is invoked by pika when RabbitMQ acknowledges the
            cancellation of a consumer. At this point we will close the channel.
            This will invoke the on_channel_closed method once the channel has been
            closed, which will in-turn close the connection.

            :param pika.frame.Method _unused_frame: The Basic.CancelOk frame
            :param str|unicode userdata: Extra user data (consumer tag)
            """
            self._consuming = False
            logger.info(f"RabbitMQ acknowledged the cancellation of the consumer: {userdata}")
            self.close_channel()

        def close_channel(self):
            """Call to close the channel with RabbitMQ cleanly by issuing the
            Channel.Close RPC command.
            """
            logger.info("Closing the channel")
            self._channel.close()

        def run(self):
            """Run the example consumer by connecting to RabbitMQ and then
            starting the IOLoop to block and allow the SelectConnection to operate.
            """
            if self._executor_stopped:
                self._executor = ThreadPoolExecutor()
                self._executor_stopped = False
            self._connection = self.connect()
            self._connection.ioloop.start()

        def stop(self):
            """Cleanly shutdown the connection to RabbitMQ by stopping the consumer
            with RabbitMQ. When RabbitMQ confirms the cancellation, on_cancelok
            will be invoked by pika, which will then closing the channel and
            connection. The IOLoop is started again because this method is invoked
            when CTRL-C is pressed raising a KeyboardInterrupt exception. This
            exception stops the IOLoop which needs to be running for pika to
            communicate with RabbitMQ. All the commands issued prior to starting
            the IOLoop will be buffered but not processed.
            """
            if not self._closing:
                self._closing = True
                logger.info(f"Stopping")
                if self._consuming:
                    self.stop_consuming()
                    self._connection.ioloop.start()
                else:
                    self._connection.ioloop.stop()
                self._executor.shutdown()
                self._executor_stopped = True
                logger.info(f"Stopped")


parse_app_properties(globals(), config.paths.config.rabbit)

# Query parameters
PY_RMQ_SERVER = RMQ_SERVER
PY_RMQ_PORT = parse_to_type(RMQ_PORT, int)
PY_RMQ_VHOST = RMQ_VHOST
PY_RMQ_USERNAME = RMQ_USERNAME
PY_RMQ_PASSWORD = RMQ_PASSWORD
PY_RMQ_HEARTBEAT_IN_SEC = parse_to_type(RMQ_HEARTBEAT_IN_SEC, int)
PY_RMQ_EXCHANGE = RMQ_EXCHANGE
PY_RMQ_QUEUE = RMQ_QUEUE
PY_RMQ_LISTEN_CONTINUOUSLY = json.loads(str(RMQ_LISTEN_CONTINUOUSLY).lower())

R_SENDER_KEY = 'senderApplication'
R_RECEIVER_KEY = 'receiverCode'
R_BUSINESS_TYPE_KEY = 'businessType'
EDX_FIELDS = ['baMessageID', 'baCorrelationID', 'fileExtension', R_BUSINESS_TYPE_KEY, R_SENDER_KEY, R_RECEIVER_KEY]


def flatten_dict_by_key(input_dict: dict, key_name, output_dict: dict = None, keep_original: bool = False):
    """
    For flattening and filtering the dict by keyword recursively

    :param input_dict: input dictionary
    :param key_name: key by which the filtering goes
    :param output_dict: output dictionary
    :param keep_original: keep original key-pairs
    :return: dictionary containing key_name and its values as list
    """
    if output_dict is None:
        output_dict = {}
    for key_item, value_item in input_dict.items():
        if keep_original:
            output_dict[key_item] = value_item
        if key_item == key_name:
            if not key_item in output_dict.keys():
                output_dict[key_item] = value_item
            else:
                if not isinstance(output_dict[key_item], list):
                    output_dict[key_item] = [output_dict[key_item]]
                output_dict[key_item].append(value_item)
        elif isinstance(value_item, dict):
            output_dict = flatten_dict_by_key(input_dict=value_item, key_name=key_name, output_dict=output_dict)

    return output_dict


def get_or_generate_communicator(address_name=None, communicators: MessageCommunicator | list = None):
    """

    :param address_name:
    :param communicators:
    :return:
    """
    communicators = communicators or AVAILABLE_SENDERS
    communicators = communicators if isinstance(communicators, Iterable) else [communicators]
    if isinstance(address_name, str) and address_name is not None:
        new_instance = next(iter([x for x in communicators if x.value_of(address_name)]), None)
        if new_instance is None:
            new_name = ''.join(choice(ascii_uppercase) for _ in range(15))
            new_instance = MessageCommunicator(mRID=new_name, receiver=address_name)
        return new_instance
    return address_name


def string_for_edx(input_str):
    """
    EDX only allows alphanumeric characters, hyphen and in some cases @ also. This escapes all other characters

    :param input_str: string to be modified
    :return: cleaned string
    """
    return re.sub(r'[^a-zA-Z0-9@]', '-', input_str)

TYPES_TO_STRINGS = {ValueOfEnum: 'value', MessageCommunicator: 'receiver'}

def to_string(input_value = None, types_values: dict = None):
    """
    Converts input value to string. if objects and fields are specified (type_values). Uses these instead

    :param input_value: value to be converted to string
    :param types_values: dictionary for the objects
    :return: converted string
    """
    types_values = types_values or TYPES_TO_STRINGS
    if not input_value:
        return input_value
    output = input_value
    for key, value in types_values.items():
        if isinstance(input_value, key) and hasattr(key, value):
            output = getattr(input_value, value)
            break
    return str(output) if output is not None else None


@dataclass
class EDXHeader:
    """

    """
    senderApplication: str | MessageCommunicator | object = None
    receiverCode: str | MessageCommunicator | object = None
    fileExtension: str | OutputFileType = None
    baMessageID: str = None
    baCorrelationID: str = None
    businessType: str = None
    additional_content: dict = None

    def _set_to_types(self):
        self.senderApplication = get_or_generate_communicator(address_name=self.senderApplication)
        self.receiverCode = get_or_generate_communicator(address_name=self.receiverCode)
        if self.fileExtension is not None and isinstance(self.fileExtension, str):
            try:
                self.fileExtension = OutputFileType.value_of(self.fileExtension)
            except ValueError:
                pass

    def __post_init__(self):
        self.update_from_additional_content()
        self._set_to_types()

    def update_from_additional_content(self, input_dict: dict = None):
        self.additional_content = {} if self.additional_content is None else self.additional_content
        input_dict = {} if input_dict is None else input_dict
        input_dict = {**self.additional_content, **input_dict}
        if isinstance(self.fileExtension, OutputFileType):
            input_dict = filter_dict_by_enum(input_dict=input_dict, enum_value=self.fileExtension)
        cleaned_dict = {}
        for k, v in input_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                cleaned_dict[k] = v
        self.additional_content = cleaned_dict

    def to_dict(self):
        output_dict = {k: to_string(input_value=getattr(self, k))
                       for k, v in self.__dict__.items() if k != 'additional_content' and v is not None}
        if self.additional_content is not None:
            output_dict = {**output_dict, **self.additional_content}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        return output_dict

    def headers_to_edx(self):
        cleaned_dict = {k: v for k, v in self.to_dict().items() if k in EDX_FIELDS}
        cleaned_dict = {k: string_for_edx(v) for k, v in cleaned_dict.items()}
        return cleaned_dict


def flatten_tuple(data):
    """
    Flattens the nested tuple to eventually a single level tuple.
    Use this when passing args as is from one handler to another

    :param data: tuple of arguments
    :return: levelled tuple

    """
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten_tuple(data[0]) + flatten_tuple(data[1:])
    else:
        return (data,)


def eval_by_type_str(input_val: Any, annotation: str = 'bytes') -> bool:
    """
    Checks if the type of input is any of the given

    :param input_val: some input that needs to be checked
    :param annotation: type annotation. For multiple separate them with |
    :return: true if any annotation matched
    """
    class_name = input_val.__class__.__name__
    if '|' in annotation:
        for single_annotation in annotation.split('|'):
            if single_annotation.strip() == class_name:
                return True
        return False
    return annotation.strip() == class_name


def eval_by_type(input_val: Any, annotation: Any) -> bool:
    """
    Checks input types. Multiple as string or single defined

    :param input_val: some input that needs to be checked
    :param annotation: type annotation. String or actual type.
    :return: true if any annotation matched
    """
    if isinstance(annotation, str):
        return eval_by_type_str(input_val=input_val, annotation=annotation)
    return isinstance(input_val, annotation)


def get_by_type_from_args(*args, output_type: Any = 'bytes', return_first: bool = True):
    """
    Gets all values by types from args.
    Note that in most cases the payload of rabbit is bytes. Need to verify if other types can be also

    :param args: input arguments
    :param output_type: annotation for filtering
    :param return_first: set it true if only first is wanted
    :return: content or None
    """
    payload = None
    args = flatten_tuple(args)
    if isinstance(args, tuple):
        items = [x for x in args if eval_by_type(x, output_type)]
        if items is not None and len(items) >= 1:
            payload = items[0] if return_first else items
    else:
        logger.warning(f"{len(args)} values detected")
    return payload


def get_message_ack_from_args(*args):
    """
    For getting MessageACK from args (used here to bypass elements in pipeline and skipping ACK)

    :param args: input arguments
    :return: MessageACK instance or None
    """
    return get_by_type_from_args(*args, output_type=MessageACK)


def get_bytes_from_args(*args):
    """
    For getting payload (content) from input arguments

    :param args: input arguments
    :return: payload if found, None otherwise
    """
    return get_by_type_from_args(*args, output_type='bytes')


class BodyHandlerInterface:

    """
    Use interface when handlers are needed to be created

    :param filter_dict: pass key value parameters to check headers if message is meant for this handler
    """

    def __init__(self, filter_dict):
        """
        Constructor

        """
        self.filter_dict = filter_dict
        self.flatten_filter_dict()

    def flatten_filter_dict(self, dict_to_flatten: dict = None):
        """
        Flattens headers (to businessType)

        :param dict_to_flatten: headers dictionary
        :return: None
        """
        dict_to_flatten = dict_to_flatten or self.filter_dict
        self.filter_dict = flatten_dict_by_key(input_dict=dict_to_flatten, key_name=R_BUSINESS_TYPE_KEY)

    def filter_message(self, **kwargs):
        """
        Filters message based on the parameters in headers.

        :param kwargs: name value pairs
        :return: True if all key-value pairs were found, false otherwise
        """
        properties = kwargs.get('properties')
        headers = {}
        if properties is not None and hasattr(properties, 'headers'):
            headers = getattr(properties, 'headers')
        if self.filter_dict and self.filter_dict != {}:
            for key, value in self.filter_dict.items():
                if key not in headers.keys():
                    return None
                value = [value] if not isinstance(value, list) else value
                if headers[key] not in value:
                    return None
        if headers is not None:
            return  check_dict_to_dataclass(EDXHeader, headers)
        else:
            return headers

    def handle(self, *args, **kwargs):
        """
        Extend this to create message handle methods

        :param args: list of parameters
        :param kwargs: key-value pairs
        :return:
        """
        pass


class ProcurementBlockingClient(BlockingClient):
    """
    ATCs arrive once a day so put up a continuous listener

    :param host: name of the host
    :param port: number of port
    :param username: username
    :param password: password
    :param queue: queue to listen
    :param message_converter: message converter (currently only 1 allowed)
    :param message_handlers: Message handlers (NB! extend BodyHandlerInterface)
    """

    def __init__(self,
                 host: str = PY_RMQ_SERVER,
                 port: int = PY_RMQ_PORT,
                 username: str = PY_RMQ_USERNAME,
                 password: str = PY_RMQ_PASSWORD,
                 queue: str = PY_RMQ_QUEUE,
                 message_converter: object | None = None,
                 message_handlers: List[BodyHandlerInterface] | None = None):
        """
        Constructor
        """
        self.queue = queue
        self.message_handlers = message_handlers
        single_message_handler = None
        if message_handlers and isinstance(message_handlers, List) and len(message_handlers) > 0:
            single_message_handler = message_handlers[0]
        super().__init__(host=host,
                         port=port,
                         username=username,
                         password=password,
                         message_converter=message_converter,
                         message_handler=single_message_handler)


    def get_message_count(self, queue: str = None):
        """
        Gets message count from queue

        :param queue: name of the queue
        :return: 0 if no messages, positive if messages, -1 if something went wrong
        """
        try:
            queue = queue or self.queue
            queue = self.consume_channel.queue_declare(queue=queue, passive=True)
            return queue.method.message_count
        except AttributeError:
            return -1


    def acknowledge_message(self, delivery_tag):
        """Acknowledge the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.
        Borrowed from rabbit.py RMQConsumer class

        :param int delivery_tag: The delivery tag from the Basic.Deliver frame
        """
        logger.info(f"Acknowledging message {delivery_tag}")
        self.consume_channel.basic_ack(delivery_tag)


    def process_message(self, basic_deliver, properties, body):
        """
        Processes input message, can use multiple handlers as pipeline. Borrowed from rabbit.py RMQConsumer class

        :param basic_deliver:
        :param properties: properties (headers)
        :param body: Message content itself
        :return:
        """
        ack = True
        message_state = None
        if self.message_handlers:
            for message_handler in self.message_handlers:
                try:
                    logger.info(f"Handling message with handler: {message_handler.__class__.__name__}")
                    body = message_handler.handle(body, properties=properties)
                    message_state = get_message_ack_from_args(body)
                    if message_state is not None:
                        ack = False
                        break
                except Exception as error:
                    logger.error(f"Message handling failed: {error}", exc_info=True)
                    ack = False
                    break
        if ack or (message_state is not None and message_state == MessageACK.ACK):
            self.acknowledge_message(basic_deliver.delivery_tag)


    def consume_until_empty(self, queue_name: str = None):
        """
        One-time function. Connect to queue and consumes the messages until there is something. If queue is

        empty then exits
        :param queue_name: name of the queue
        :return: None
        """
        queue_name = queue_name or self.queue
        while True:
            message_count = self.get_message_count(queue=queue_name)
            if message_count <= 0:
                logger.info(f"{queue_name} is finally empty, exiting")
                break
            logger.info(f"{message_count} messages in queue, taking one down and parsing it around...")
            method_frame, properties, body = self.get_single_message(queue=queue_name, auto_ack=False)
            if method_frame:
                self.process_message(basic_deliver=method_frame, properties=properties, body=body)
            else:
                time.sleep(0.5)
        self.close()


class ProcurementRMQConsumer(RMQConsumer):
    """
    Extended version for usage in Procurement service

    :param host: name of the host
    :param port: number of port
    :param username: username
    :param password: password
    :param heartbeat: heartbeat
    :param queue: queue to listen
    :param message_converter: message converter (currently only 1 allowed)
    :param message_handlers: Message handlers (NB! extend BodyHandlerInterface)
    """

    def __init__(self,
                 host: str = PY_RMQ_SERVER,
                 port: int = PY_RMQ_PORT,
                 vhost: str = PY_RMQ_VHOST,
                 username: str = PY_RMQ_USERNAME,
                 password: str = PY_RMQ_PASSWORD,
                 heartbeat: int = PY_RMQ_HEARTBEAT_IN_SEC,
                 queue: str = PY_RMQ_QUEUE,
                 message_handlers: list = None,
                 message_converter: object = None):
        """
        Constructor
        """
        super().__init__(host=host,
                         port=port,
                         vhost=vhost,
                         username=username,
                         password=password,
                         heartbeat=heartbeat,
                         que=queue,
                         message_handlers=message_handlers,
                         message_converter=message_converter)


def send_data_to_rabbit(input_data,
                        rabbit_exchange: str =PY_RMQ_EXCHANGE,
                        rabbit_client: BlockingClient = None,
                        headers: dict = None):
    """
    Sends data to rabbit exchange

    :param input_data: eventually str (if dictionaries, lists then json dumps)
    :param rabbit_exchange: name of the rabbit exchange
    :param rabbit_client: rabbit client
    :param headers: headers if needed
    :return:
    """

    rabbit_client = rabbit_client or ProcurementBlockingClient()
    if isinstance(input_data, BytesIO):
        input_data = input_data.getvalue()
    elif isinstance(input_data, bytes):
        pass
    elif not isinstance(input_data, str):
        input_data = json.dumps(input_data)
    # clean headers for PDN
    headers = {k: v for k, v in headers.items() if v is not None}

    rabbit_client.publish(payload=input_data,
                          exchange_name=rabbit_exchange,
                          headers=headers)