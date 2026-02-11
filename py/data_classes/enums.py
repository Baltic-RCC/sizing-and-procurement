from ast import literal_eval
from enum import Enum


class ValueOfEnum(Enum):

    def __str__(self):
        return str(self.value)

    @classmethod
    def value_of(cls, value):
        """
        Parses input to self if  value matches

        :param value: input string
        :return: found value or error
        """
        value = str(value).upper()
        for k, v in cls.__members__.items():
            if k == value:
                return v
        else:
            raise ValueError(f"'{cls.__name__}' enum not found for '{value}' ")


def parse_to_enum(input_value: str, enum, default_value = None):
    """
    Tries to parse string to enum by value

    :param input_value: input string
    :param enum: Enum type
    :param default_value: add default value if not match was found
    :return:
    """
    try:
        return enum.value_of(input_value)
    except ValueError:
        return default_value


def parse_to_enum_by_value(input_value: str, enum_type):
    """
    Workaround to get enum from string

    :param input_value: input string
    :param enum_type: Enum class
    :return: Enum if found, None otherwise
    """
    search_str = str(input_value).lower()
    for enum_name, enum_self in enum_type.__members__.items():
        # if str(enum_name).lower() in str(input_value).lower():
        if str(enum_name).lower() in search_str or str(enum_self.value).lower() in search_str:
            return enum_self
    return None


def get_enum_name_value(input_value: list | str, enum_type):
    """
    Gets first positive match for enum and returns its value and key that triggered

    :param input_value: list where to search (keys)
    :param enum_type: enum tpe
    :return: key and enum value if found, None and None otherwise
    """
    input_value = [input_value] if isinstance(input_value, str) else input_value
    for _, key_val in enumerate(input_value):
        match = parse_to_enum_by_value(input_value=key_val, enum_type=enum_type)
        if match is not None:
            return match.name, match.value
    return None, None



class NameValueOfEnum(Enum):

    @classmethod
    def value_of(cls, value):
        """
        Parses input to self if name or value matches

        :param value: input string
        :return: found value or error
        """
        try:
            value_up = str(value).upper()
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            value_up = None
        try:
            value_in_type = literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            value_in_type = None
        for k, v in cls.__members__.items():
            if value_up and str(k).upper() == value_up:
                return v
            if isinstance(v.value, str) and value_up and str(v.value).upper() == value_up:
                return  v
            if v.value == value_in_type:
                return v
        else:
            raise ValueError(f"'{cls.__name__}' enum not found for '{value}' ")

    def __str__(self):
        return self.value


class MessageACK(ValueOfEnum):
    ACK = 'ACK'
    NACK = 'NACK'


class OutputFileType(ValueOfEnum):
    """
    Selects output file format
    """
    XML = 'XML'
    XLSX = 'XLSX'
    BOTH = 'BOTH'


class ProcurementCalculationType(ValueOfEnum):
    """
    Select calculation type
    """
    NCPB = 'NCPB'       # Bids
    ATC = 'ATC'         # ATC
    ALL = 'ALL'         # Both


class ExceededPercentType(NameValueOfEnum):
    """
    For selecting percent strategy
    """
    MIN = 'min'
    MEDIAN = 'median'
    MAX = 'max'

# codes from list

class InputColumnName(NameValueOfEnum):
    """
    Input data parsing keywords
    """
    REGION_COLUMN = 'domain'
    DIRECTION_COLUMN = 'direction'
    TYPE_COLUMN = 'type'
    TYPE_DIRECTION_COLUMN = 'description'
    VALUE_COLUMN = 'value'
    START_TIME_COLUMN = 'start_time'
    END_TIME_COLUMN = 'end_time'
    REGION_IN_COLUMN = 'in_domain'
    REGION_OUT_COLUMN = 'out_domain'


class FlowDirectionType(NameValueOfEnum):
    """
    Codes for '(flow)direction' in xmls
    """

    UP = 'A01'          # Up signifies that the available power can be used by the Purchasing area to increase energy.
    DOWN = 'A02'        # Down signifies that the available power can be used by the Purchasing area to decrease energy
    UP_AND_DOWN = 'A03' # Up and Down signifies that the UP and Down values are equal.
    STABLE = 'A04'      # The direction at a given instant in time is considered to be stable

    @classmethod
    def value_of(cls, value):
        value = str(value).upper()
        for k, v in cls.__members__.items():
            v_value = str(v.value)
            if str(k) in value or v_value in value:
                return v
        else:
            raise ValueError(f"'{cls.__name__}' enum not found for '{value}' ")


class EnergyProductType(NameValueOfEnum):
    """
    Codes for 'product' in xmls
    """
    ACTIVE_POWER = '8716867000016'     # The product of voltage and the in-phase component of current measured in watts
    REACTIVE_POWER = '8716867000023'   # The product of voltage and current and the sine of the phase angle between them
    ACTIVE_ENERGY = '8716867000030'    # The electrical energy produced, measured in units of watt-hours,
    REACTIVE_ENERGY = '8716867000047'               # The integral with respect to time of reactive power
    CAPACITIVE_REACTIVE_POWER = '8716867000115'     # Capacitive reactive power.
    INDUCTIVE_REACTIVE_POWER = '8716867000122'      # Inductive reactive power.
    CAPACITIVE_REACTIVE_ENERGY = '8716867000139'    # Capacitive reactive energy.
    INDUCTIVE_REACTIVE_ENERGY = '8716867000146'     # Inductive reactive energy.
    WATER = '8716867009911'                         # For hydropower stations,


class CurveType(NameValueOfEnum):
    """
    Codes for 'CurveType' in xmls
    """
    SEQUENTIAL_FIXED_SIZE_BLOCK = 'A01'             # The curve is made of successive Intervals of time
    POINT = 'A02'                                   # The curve is made of successive instants of time (Points).


class ProcessType(NameValueOfEnum):
    """
    Codes for 'process.processType' in xmls
    """
    CAPACITY_ALLOCATION = 'A07'            # The information provided concerns the capacity allocation process
    FORECAST = 'A14'                       # The data contained the document are to be handled in forecasting process
    CAPACITY_DETERMINATION = 'A15'         # The process of determining the capacity for use.
    REALISED = 'A16'                       # The process for the treatment of realised data as opposed to forecast data.
    RESERVE_RESOURCE_PROCESS = 'A27'       # The process being described is for general reserve resources.
    CONTRACTED = 'A34'                     # The process being described is for contracted information
    MODIFICATION = 'A37'                   # The process being described is for the modification of information
    INTRADAY_PROCESS = 'A40'               # The process being described is for intraday process.
    REPLACEMENT_RESERVE = 'A46'            # A process being described is for replacement reserves (RR)
    MANUAL_FREQUENCY_RESTORATION_RESERVE = 'A47'    # A process being described is formFRR.
    INTRADAY_CAPACITY_DETERMINATION = 'A49'         # The process run at the ID timeframe to determine the capacity
    AUTOMATIC_FREQUENCY_RESTORATION_RESERVE = 'A51' # A process being described is for aFRR
    FREQUENCY_CONTAINMENT_RESERVE = 'A52'           # A process being described is for FCR.
    FREQUENCY_RESTORATION_RESERVE = 'A56'  # The process being described is for general frequency restoration reserve
    SCHEDULED_ACTIVATION_MFRR = 'A60'      # mFRR being subject to scheduled activation
    DIRECT_ACTIVATION_MFRR = 'A61'         # mFRR being subject to direct activation
    CENTRAL_SELECTION_AFRR = 'A67'         # aFRR subject to central selection of bids for activation
    LOCAL_SELECTION_AFRR = 'A68'           # aFRR subject to local selection of bids for activation


class BusinessType(NameValueOfEnum):
    """
    Codes for 'businessType' in xmls (TimeSeries)
    """
    IMBALANCE_VOLUME = 'A20'                # Imbalance between meter readings and the balance corrected with bids
    FREQUENCY_CONTROL = 'A22'               # A time series concerning primary and secondary reserve
    BALANCE_MANAGEMENT = 'A23'              # A time series concerning energy balancing services
    AVAILABLE_TRANSFER_CAPACITY = 'A26'     # Available transfer capacity for cross-border exchanges
    OFFERED_CAPACITY = 'A31'                # The time series provides the offered capacity
    FREQUENCY_CONTAINMENT_RESERVE = 'A95'   # The business being described concerns frequency containment reserve
    AUTOMATIC_FREQUENCY_RESTORATION_RESERVE = 'A96' # The business being described concerns aFRR
    MANUAL_FREQUENCY_RESTORATION_RESERVE = 'A97'    # The business being described concerns mFRR
    REPLACEMENT_RESERVE = 'A98'             # The business being described concerns replacement reserve
    AREA_CONTROL_ERROR = 'B33'              # The sum of the instantaneous difference between actual and the set-point
    OFFER = 'B74'                           # The time series provides an offer to provide reserves
    NEED = 'B75'                            # The timeseries provides a requirement for reserves
    PROCURED_CAPACITY = 'B95'               # An accepted offer of balancing capacity
    USED_CAPACITY = 'B96'                   # The used cross-zonal balancing capacity
    REMAINING_CAPACITY = 'C01'              # A time series concerning the remaining capacity
    CAPACITY_ALLOCATED = 'C19'              # The business being described concerns capacity allocation (excludes price)
    SHARE_OF_RESERVE_CAPACITY = 'C23'       # A time series concerning the share of reserve capacity
    PERCENTILE = 'C68'                      # A time series describing percentiles
    FORECASTED_CAPACITY = 'C76'             # A time series describing forecasted capacity
    MINIMUM_AVAILABLE_CAPACITY = 'C77'      # A time series describing minimum available capacity
    ENERGY_RESERVES = 'C89'                 # A timeseries describing energy reserves


class MessageType(NameValueOfEnum):
    """
    Codes for 'type' in xmls
    """
    ACQUIRING_SYSTEM_OPERATOR_RESERVE_SCHEDULE = 'A15' # A document providing reserve purchases
    BID_DOCUMENT = 'A24'                    # A Document providing bid information
    CAPACITY_DOCUMENT = 'A26'               # A document providing capacity information
    PROPOSED_CAPACITY = 'A32'               # The capacity proposed for agreement between parties
    RESERVE_TENDER_DOCUMENT = 'A37'         # The document that is used for the tendering for reservers
    RESERVE_ALLOCATION_RESULT = 'A38'       # The document used to provide the results of a Reserve auction
    CONTRACTED_RESERVES = 'A81'             # A document providing the reserves contracted for a period
    ACCEPTED_OFFERS = 'A82'                 # A document providing the offers of reserves that have been accepted
    ACTIVATED_BALANCING_QUANTITIES = 'A83'  # A document providing the quantities of reserves that have been activated
    IMBALANCE_VOLUME = 'A86'                # A document providing the volume of the imbalance for a period
    CROSS_BORDER_BALANCING = 'A88'          # A document providing the cross border balancing requirements for a period
    IMBALANCE_PROGNOSIS = 'B39'             # A document to provide the prognosis of energy imbalances for a region
    BID_AVAILABILITY = 'B45'                # Providing the reasons for changing the availability or volume of a bid.


class MeasurementUnitType(NameValueOfEnum):
    """
    Codes for 'measurementUnit.name' in xmls
    """
    GIGAWATT = 'A90'                    # GW unit as per UN/CEFACT recommendation 20.
    ONE = 'C62'                         # A unit for dimensionless quantities, also called quantities of dimension one.
    GIGAWATT_HOUR = 'GWH'               # GWh unit as per UN/CEFACT recommendation 20.
    KILOVOLT_AMPERE_REACTIVE = 'KVR'    # A unit of electrical reactive power
    KILOWATT_HOUR = 'KWH'               # A total amount of electrical energy transferred or consumed in one hour
    KILOWATT = 'KWT'                    # A unit of bulk power flow, which can be defined as the rate of energy transfer
    MEGAVOLT_AMPERE_REACTIVE_HOURS = 'MAH'  # Total amount of reactive power across a power system.
    MEGAVOLT_AMPERE_REACTIVE = 'MAR'    # A unit of electrical reactive power by a current of one thousand amperes
    MEGAWATT = 'MAW'                    # A unit of bulk power flow, which can be defined as the rate of energy transfer
    MEGAWATT_HOURS = 'MWH'              # The total amount of bulk energy transferred or consumed.
    PERCENT = 'P1'                      # A unit of proportion equal to 0.01.
    WATT = 'WTT'                        # The watt is the International System of Units


class CodingSchemeType(NameValueOfEnum):
    """
    Codes for 'codingScheme' in xmls
    """
    EIC = 'A01'                         # The coding scheme is the Energy Identification Coding Scheme (EIC)


class RoleType(NameValueOfEnum):
    """
    Codes for '...role.Type' in xmls
    """
    SYSTEM_OPERATOR = 'A04'
    CONTROL_AREA_OPERATOR = 'A14'
    CONTROL_BLOCK_OPERATOR = 'A15'
    COORDINATION_CENTRE_OPERATOR = 'A16'
    CAPACITY_COORDINATOR = 'A36'    # A party, responsible for establishing a coordinated Offered Capacity, NTC and ATC
    DATA_PROVIDER = 'A39'           # A party that is responsible for providing information to a central authority.
    REGIONAL_SECURITY_COORDINATOR = 'A44'   # The RSC as defined in the System Operation guideline.
    LFC_OPERATOR = 'A48'            # A party responsible for the Load Frequency Control of its LFC Area or block.
    TRANSMISSION_SYSTEM_OPERATOR = '"49'
    RESOURCE_AGGREGATOR = 'A52' # A party that aggregates resources for usage by a service provider for energy services.
    FLEXIBILITY_SERVICE_PROVIDER = 'A56' # A party that offers flexibility services based on acquired resources.


class StatusType(NameValueOfEnum):
    """
    Codes for 'doc.Status' in xmls
    """
    INTERMEDIATE = 'A01'    # The document is in a non finalized state.
    FINAL = 'A02'           # The document is in a definitive state
    AVAILABLE = 'A06'       # The volumes (one or more) are available.
    ACTIVATED = 'A07'       # The quantities in the time series have been activated.
    IN_PROGRESS = 'A08'     # The quantities in the time series are in the process of activation
    ORDERED = 'A10'         # The quantities in the time series are to be activated.
    RESULT = 'A32'          # Result
    CONFIRMED = 'A37'       # The status is confirmed.
    SHALL_BE_USED = 'A38'   # The object defined in the series shall be used
    COULD_BE_USED = 'A39'   # The object defined in the series could be used.
    PROPOSED = 'A40'        # The status of the information is proposed.


class MarketProductType(NameValueOfEnum):
    """
    Codes for Standard_MarketProduct
    """
    STANDARD_BALANCING_PRODUCT = 'A01'
    MFRR_PRODUCT_FOR_SCHEDULED_ACTIVATION = 'A05'
    MFRR_PRODUCT_FOR_DIRECT_ACTIVATION = 'A06'
    MFRR_PRODUCT_FOR_SCHEDULED_DIRECT_ACTIVATION = 'A07'


class IndicatorType(NameValueOfEnum):
    """
    Boolean for xmls
    """
    YES = 'A01'
    NO = 'A02'


class EICCodeType(NameValueOfEnum):
    """
    For categorizing EIC codes
    """
    RCC = 'Coordination Center Operator'
    TSO = 'System Operator'
    LFC_AREA = 'LFC Area'
    LFC_BLOCK = 'LFC Block'


class BusinessProductType(NameValueOfEnum):
    """
    Select BusinessType by product
    """
    mFRR = ProcessType.MANUAL_FREQUENCY_RESTORATION_RESERVE.value
    aFRR = ProcessType.AUTOMATIC_FREQUENCY_RESTORATION_RESERVE.value
    SA_mFRR = ProcessType.SCHEDULED_ACTIVATION_MFRR.value
    DA_mFRR = ProcessType.DIRECT_ACTIVATION_MFRR.value
    CA_aFRR = ProcessType.CENTRAL_SELECTION_AFRR.value
    LA_aFRR = ProcessType.LOCAL_SELECTION_AFRR.value
    ATC = 'ATC'


class BidDirection(NameValueOfEnum):
    """
    For parsing Upward and Downward
    """
    Upward = FlowDirectionType.UP.value
    Downward = FlowDirectionType.DOWN.value


class NegativeValuesHandler(NameValueOfEnum):
    """
    How to deal negative values
    """
    CUT_NEGATIVE = 'Cut negative'
    NEGATIVE_TO_ZERO = 'Negative to zero'
    DO_NOTHING = 'NONE'


class ExceededEnumOperator(NameValueOfEnum):
    """
    How to calculate exceeded values
    """
    CASCADE = 'CASCADE'
    PROPORTION = 'PROPORTION'
    DO_NOTHING = 'NONE'
