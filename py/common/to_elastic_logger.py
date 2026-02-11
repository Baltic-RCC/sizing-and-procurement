import sys
import logging
import requests
import config
from py.common.config_parser import parse_app_properties
from py.common.functions import str_to_bool

from py.handlers.elastic_handler import PY_ELASTICSEARCH_HOST, PY_SIZING_AND_PROCUREMENT_LOGS_INDEX, Elastic

# Root logger
root_logger = logging.getLogger()

# Local logger
logger = logging.getLogger(__name__)
parse_app_properties(globals(), config.paths.config.to_elastic_logger)
PY_LOGGING_FORMAT = LOGGING_FORMAT
PY_LOGGING_DATE_FORMAT = LOGGING_DATE_FORMAT
PY_LOGGING_LEVEL = LOGGING_LEVEL
PY_LOGGING_TO_ELASTIC = str_to_bool(str(LOGGING_TO_ELASTIC).lower())

logging.basicConfig(
    format=PY_LOGGING_FORMAT,
    datefmt=PY_LOGGING_DATE_FORMAT,
    level=PY_LOGGING_LEVEL,
    handlers=[logging.StreamHandler(sys.stdout)]
)

class ElkLoggingHandler(logging.StreamHandler):
    """
    Handler for sending log entries to elastic

    :param elastic_server: Address of elastic server
    :param elastic_index:  Index where to store the logs
    :param logging_level: Level to be logged
    :param logging_format: Log line format for log entries
    :param logging_date_format: Date format for log entries
    :param extra_fields: Add extra fields to messages
    :param fields_filter: Filters logs by the fields
    """

    def __init__(self,
                 elastic_server: str = PY_ELASTICSEARCH_HOST,
                 elastic_index: str = PY_SIZING_AND_PROCUREMENT_LOGS_INDEX,
                 logging_level: str = PY_LOGGING_LEVEL,
                 logging_format: str = PY_LOGGING_FORMAT,
                 logging_date_format: str = PY_LOGGING_DATE_FORMAT,
                 extra_fields: dict | None = None,
                 fields_filter: list | None = None):
        """
        Constructor
        """
        super().__init__(sys.stdout)
        self.server = elastic_server
        self.index = elastic_index
        self.extra_fields = dict()
        if extra_fields:
            self.extra_fields = extra_fields

        self.fields_filter = fields_filter
        self.connected = self.elk_connection()

        # Set level and format from settings
        self.setLevel(logging_level)
        formatter = logging.Formatter(fmt=logging_format, datefmt=logging_date_format)
        self.setFormatter(formatter)

    def elk_connection(self):
        """
        Checks connection to elastic.

        :return: None
        """
        try:
            response = requests.get(self.server, timeout=5)
            if response.status_code == 200:
                logger.info(f"Connection to {self.server} successful")
                return True
            else:
                logger.warning(f"ELK server response: [{response.status_code}] {response.reason}. Disabling ELK logging.")
        except requests.exceptions.ConnectTimeout:
            logger.warning(f"ELK server {self.server} does not responding with ConnectTimeout error. Disabling ELK logging.")
        except Exception as e:
            logger.warning(f"ELK server {self.server} returned unknown error: {e}")

    def elk_formatter(self, record):
        """
        Filters message for elastic

        :param record: input message
        :return: filtered message
        """
        elk_record = record.__dict__
        if self.fields_filter:
            elk_record = {key: elk_record[key] for key in self.fields_filter if key in elk_record}

        return elk_record

    def emit(self, record):
        """
        Sends message to elastic

        :param record: input message
        :return: None
        """
        elk_record = self.elk_formatter(record=record)

        # Add extra global attributes from class initiation
        if self.extra_fields:
            elk_record.update(self.extra_fields)
        # Send to Elk
        Elastic.send_to_elastic(index=self.index, json_message=elk_record, server=self.server)


def initialize_custom_logger(
        logging_level: str = PY_LOGGING_LEVEL,
        logging_format: str = PY_LOGGING_FORMAT,
        logging_date_format: str = PY_LOGGING_DATE_FORMAT,
        elastic_server: str = PY_ELASTICSEARCH_HOST,
        elastic_index: str = PY_SIZING_AND_PROCUREMENT_LOGS_INDEX,
        logging_to_elastic: bool = PY_LOGGING_TO_ELASTIC,
        extra_fields: None | dict = None,
        fields_filter: None | list = None,
        ):
    """
    Initializes gathering the logs to elastic

    :param elastic_server: Address of elastic server
    :param elastic_index:  Index where to store the logs
    :param logging_level: Level to be logged
    :param logging_format: Log line format for log entries
    :param logging_date_format: Date format for log entries
    :param extra_fields: Add extra fields to messages
    :param fields_filter: Filters logs by the fields
    :param logging_to_elastic: If true then starts to gather logs to elastic
    :return: logger instance or None
    """
    if not logging_to_elastic:
        logger.info(f"Logging to elastic switched off: {logging_to_elastic}")
        return None
    logger.info(f"Start gathering logs to elastic")
    root_logger.setLevel(logging_level)
    root_logger.propagate = True

    # Configure Elk logging handler
    elastic_log_handler = ElkLoggingHandler(elastic_server=elastic_server,
                                            elastic_index=elastic_index,
                                            extra_fields=extra_fields,
                                            fields_filter=fields_filter,
                                            logging_level=logging_level,
                                            logging_format=logging_format,
                                            logging_date_format=logging_date_format)

    if elastic_log_handler.connected:
        root_logger.addHandler(elastic_log_handler)
    else:
        logger.warning(f"Elk logging handler not initialized")
    return elastic_log_handler


if __name__ == '__main__':
    # Start root logger
    STREAM_LOG_FORMAT = "%(levelname) -10s %(asctime) -10s %(name) -35s %(funcName) -30s %(lineno) -5d: %(message)s"
    logging.basicConfig(stream=sys.stdout,
                        format=STREAM_LOG_FORMAT,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        )

    # Test ELK custom logger
    elk_handler = ElkLoggingHandler()
    if elk_handler.connected:
        logger.addHandler(elk_handler)
    logger.info(f"Info message", extra={'extra': 'logger testing'})
    print("Done")
