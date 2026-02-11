import logging
import sys
from datetime import datetime

import config
from py.common.config_parser import parse_app_properties
from py.common.functions import convert_input, calculate_start_and_end_date
from py.common.df_functions import parse_dataframe_to_nested_dict
from py.common.time_functions import convert_datetime_to_string, convert_datetime_to_string_utc, get_time_intervals
from py.handlers.elastic_handler import get_data_from_elastic_by_time, PY_PROCUREMENT_PROD_ATC_INDEX, \
    dict_to_and_or_query
from py.handlers.rabbit_handler import send_data_to_rabbit, ProcurementBlockingClient, PY_RMQ_EXCHANGE

logger = logging.getLogger(__name__)

parse_app_properties(globals(), config.paths.config.atc_input)
parse_app_properties(globals(), config.paths.config.rabbit)

# Query parameters
PY_PROD_ATC_TIME_KEY = PROD_ATC_TIME_KEY
PY_PROD_ATC_PARAMETERS = convert_input(PROD_ATC_PARAMETERS)
PY_PROD_ATC_BINDING_ARGUMENTS = convert_input(PROD_ATC_BINDING_ARGUMENTS)

# Query time intervals
PY_PROD_ATC_QUERY_START_DATE = PROD_ATC_QUERY_START_DATE
PY_PROD_ATC_QUERY_END_DATE = PROD_ATC_QUERY_END_DATE
PY_PROD_ATC_TIMEDELTA = PROD_ATC_TIMEDELTA
PY_PROC_ATC_OFFSET = PROC_ATC_OFFSET

ELASTIC_MAX_DAYS = 'P30D'


def query_data_from_elastic(query_start_time: str | datetime,
                            query_end_time: str | datetime,
                            query_index: str = PY_PROCUREMENT_PROD_ATC_INDEX,
                            query_parameters: dict = PY_PROD_ATC_PARAMETERS,
                            query_time_key: str = PY_PROD_ATC_TIME_KEY,
                            use_default_fields: bool = False):
    """
    Queries data from elastic (ATC)

    :param query_start_time: start time from where to start query
    :param query_end_time: end time to when to query
    :param query_index: atc index
    :param query_parameters: additional parameters (process type etc.)
    :param query_time_key: time key to be used
    :param use_default_fields: whether to unpack fields (useful when nested dicts are used)
    :return: dataframe with data
    """
    query_for_data = dict_to_and_or_query(value_dict=query_parameters, key_name='match')
    # keep in mind that change use_default_fields to True when nested dicts start to come in
    received_data = get_data_from_elastic_by_time(start_time_value=query_start_time,
                                                  end_time_value=query_end_time,
                                                  elastic_index=query_index,
                                                  elastic_query=query_for_data,
                                                  use_default_fields=use_default_fields,
                                                  time_interval_key = query_time_key)
    return received_data


if __name__ == '__main__':

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    start_date, end_date = calculate_start_and_end_date(start_date_time=PY_PROD_ATC_QUERY_START_DATE,
                                                        end_date_time=PY_PROD_ATC_QUERY_END_DATE,
                                                        offset=PY_PROC_ATC_OFFSET,
                                                        time_delta=PY_PROD_ATC_TIMEDELTA)
    date_pairs = get_time_intervals(start_date_value=start_date, end_date_value=end_date, time_delta=ELASTIC_MAX_DAYS)
    send_message = True
    if send_message:
        rabbit_service = ProcurementBlockingClient()
    else:
        rabbit_service = None
    logger.info(f"Querying from {convert_datetime_to_string_utc(start_date)} to "
                f"{convert_datetime_to_string_utc(end_date)} in {len(date_pairs)} intervals")
    for date_pair in date_pairs:
        start_time, end_time = date_pair
        logger.info(f"Querying from {convert_datetime_to_string(start_time)} to {convert_datetime_to_string(end_time)}")
        elastic_data = query_data_from_elastic(query_start_time=start_time,
                                               query_end_time=end_time,
                                               query_index=PY_PROCUREMENT_PROD_ATC_INDEX,
                                               query_parameters=PY_PROD_ATC_PARAMETERS,
                                               query_time_key=PY_PROD_ATC_TIME_KEY,
                                               use_default_fields=False)
        nested_dicts = parse_dataframe_to_nested_dict(input_dataframe=elastic_data)

        special_headers = {"Source": PY_PROCUREMENT_PROD_ATC_INDEX,
                           'time_key': PY_PROD_ATC_TIME_KEY,
                           'from': convert_datetime_to_string_utc(start_time),
                           'to': convert_datetime_to_string_utc(end_time)}
        if PY_PROD_ATC_BINDING_ARGUMENTS and isinstance(PY_PROD_ATC_BINDING_ARGUMENTS, dict):
            special_headers = {**PY_PROD_ATC_BINDING_ARGUMENTS, **special_headers, **PY_PROD_ATC_PARAMETERS}
        else:
            special_headers = {**special_headers, **PY_PROD_ATC_PARAMETERS}

        if rabbit_service is not None:
            logger.info(f"Sending to {PY_RMQ_EXCHANGE}")
            logger.info(f"headers: {special_headers}")
            send_data_to_rabbit(input_data=nested_dicts,
                                rabbit_exchange=PY_RMQ_EXCHANGE,
                                rabbit_client=rabbit_service,
                                headers=special_headers)
    print("Done")



