# FOLLOWING IS TAKEN FROM EMF_OS/emf/common/config_parser.py

import configparser
import logging
import os
import ast
from typing import Dict, Any

logger = logging.getLogger(__name__)


def parse_app_properties(caller_globals: Dict[str, Any],
                         path: str,
                         section: str = "MAIN",
                         sanitize_mask: str = "****",
                         eval_types: bool = False) -> None:
    """
    Parses application properties and assigns values to caller_globals dictionary

    :param caller_globals: The dictionary of caller's globals to assign the parsed properties
    :param path: The path to the .properties file
    :param section: The section name in the properties file to parse. Defaults to "MAIN"
    :param sanitize_mask:  The mask to use for sanitizing sensitive values
    :param eval_types: Flag to convert strings to native datatypes. Defaults to False
    :return:
    """
    # Configure settings parser
    raw_settings = configparser.RawConfigParser()
    raw_settings.optionxform = str

    # Load settings
    raw_settings.read(path)

    for setting in raw_settings.items(section):

        # Get parameter name and value from settings
        parameter_name, parameter_config_value = setting

        # Force parameter name to upper, to follow python PEP and also force that all settings are defined with
        # capital letters
        parameter_name = parameter_name.upper()

        # Get parameter value from ENV, if available
        parameter_env_value = os.getenv(parameter_name)

        # Check if password needs to be sanitized
        sanitize = "PASSWORD" in parameter_name

        # If parameter is defined in ENV
        if parameter_env_value:
            defined_in = "ENVIRONMENT"
            parameter_value = parameter_env_value

        # If not, take the default value form config
        else:
            defined_in = "PROPERTIES"
            parameter_value = parameter_config_value

        # Sanitize parameter value for logging
        sanitized_parameter_value = sanitize_mask if sanitize else parameter_value

        logger.info(f"[{defined_in}] {parameter_name} = {sanitized_parameter_value}",
                    extra={"parameter_defined_in": defined_in,
                           "parameter_name": parameter_name,
                           "parameter_value": sanitized_parameter_value})

        # Convert string parameters to native datatypes
        if eval_types:
            try:
                parameter_value = ast.literal_eval(parameter_value)
            except ValueError:
                pass

        # Assign value to globals with upper letters
        caller_globals[parameter_name] = parameter_value


# TEST
if __name__ == "__main__":

    import sys
    import config

    logging.basicConfig(
        format='%(levelname)-10s %(asctime)s.%(msecs)03d %(name)-30s %(funcName)-35s %(lineno)-5d: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    os.environ["OPDM_PASSWORD"] = "1"

    parse_app_properties(globals(), config.paths.opdm_integration.opdm)
