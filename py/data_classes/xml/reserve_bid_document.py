from dataclasses import dataclass
from datetime import datetime

from lxml import etree

from py.common.functions import dict_to_dataclass
from py.common.time_functions import convert_datetime_to_string_utc, time_delta_to_str
from py.data_classes.elastic.elastic_data_models import DataToXMLPoint
from py.data_classes.enums import IndicatorType
from py.data_classes.task_classes import EICCodeOwner, EICParty
from py.data_classes.xml.base_xml import XMLDocument, XML_DATE_FORMAT, XMLBidTimeSeries, XML_TO_RESULT_MAP
from py.common.ref_constants import POINT_RESOLUTION_KEY, AVAILABLE_POSITION_KEY, AVAILABLE_VALUE_KEY, \
    PERCENTAGE_VALUE_KEY, MRID_KEY, LFC_BLOCK_MRID_KEY, BUSINESS_TYPE_KEY, DOMAIN_MRID_KEY, CURVE_TYPE_KEY, \
    DIRECTION_CODE_KEY, PERCENTAGE_UNIT_KEY, AVAILABLE_UNIT_KEY, SERIES_VALID_TO_KEY, SERIES_VALID_FROM_KEY, \
    VALID_FROM_KEY, VALID_TO_KEY

RESERVE_BID_DOCUMENT_NAME = "ReserveBid_MarketDocument"
RESERVE_XML_TO_RESULT_MAP = {'Bid_TimeSeries.acquiring_Domain.mRID': LFC_BLOCK_MRID_KEY,
                             'Bid_TimeSeries.businessType': BUSINESS_TYPE_KEY,
                             'Bid_TimeSeries.mRID': MRID_KEY,
                             'Bid_TimeSeries.connecting_Domain.mRID': DOMAIN_MRID_KEY,
                             'Bid_TimeSeries.curveType': CURVE_TYPE_KEY,
                             'Bid_TimeSeries.flowDirection.direction': DIRECTION_CODE_KEY,
                             'Bid_TimeSeries.price_Measurement_Unit.name': PERCENTAGE_UNIT_KEY,
                             'Bid_TimeSeries.quantity_Measurement_Unit.name': AVAILABLE_UNIT_KEY,
                             'Period.resolution': POINT_RESOLUTION_KEY,
                             'Point.minimum_Quantity.quantity': PERCENTAGE_VALUE_KEY,
                             'Point.position': AVAILABLE_POSITION_KEY,
                             'Point.quantity.quantity': AVAILABLE_VALUE_KEY,
                             'Period.timeInterval.end': SERIES_VALID_TO_KEY,
                             'Period.timeInterval.start': SERIES_VALID_FROM_KEY,
                             'utc_start': VALID_FROM_KEY,
                             'utc_end': VALID_TO_KEY,
                             }

RESERVE_XML_TO_RESULT_MAP = {**XML_TO_RESULT_MAP, **RESERVE_XML_TO_RESULT_MAP}


@dataclass
class ReserveBidXMLTimeSeries(XMLBidTimeSeries):

    def __init__(self, points: DataToXMLPoint | list = 1):
        """
        Initializes ReserveBid Timeseries

        :param points: how many points to be added
        """
        super().__init__(points=points,
                         element_name='Bid_TimeSeries',
                         point_primary_field='quantity.quantity',
                         point_secondary_field='minimum_Quantity.quantity')
        self.mrid_element = etree.SubElement(self.main_element, "mRID")
        self.auction = etree.SubElement(self.main_element, "auction.mRID")
        self.business_type = etree.SubElement(self.main_element, "businessType")
        self.acquiring_domain = etree.SubElement(self.main_element, "acquiring_Domain.mRID", codingScheme="A01")
        self.connecting_domain = etree.SubElement(self.main_element, "connecting_Domain.mRID", codingScheme="A01")
        # self.provider = etree.SubElement(self.main_element, "provider_MarketParticipant.mRID", codingScheme="A01")
        self.measurement_unit = etree.SubElement(self.main_element, "quantity_Measurement_Unit.name")
        # self.currency_unit = etree.SubElement(self.main_element, "currency_Unit.name")
        self.price_measurement_unit = None
        if any([k.secondary_quantity for k in self.points]):
            self.price_measurement_unit = etree.SubElement(self.main_element, "price_Measurement_Unit.name")
        self.divisible = etree.SubElement(self.main_element, "divisible")
        # self.linked_bids_identification = etree.SubElement(self.main_element, "linkedBidsIdentification")
        # self.multipart_bid_identification = etree.SubElement(self.main_element, "multipartBidIdentification")
        # self.exclusive_bids_identification = etree.SubElement(self.main_element, "exclusiveBidsIdentification")
        # self.block_bid = etree.SubElement(self.main_element, "blockBid")
        # self.status = etree.SubElement(self.main_element, "status")
        # self.priority = etree.SubElement(self.main_element, "priority")
        self.flow_direction = etree.SubElement(self.main_element, "flowDirection.direction")
        # self.step_increment_quantity = etree.SubElement(self.main_element, "stepIncrementQuantity")
        # self.energy_price_measurement_unit = etree.SubElement(self.main_element, "energyPrice_Measurement_Unit.name")
        # self.market_agreement_type = etree.SubElement(self.main_element, "marketAgreement.type")
        # self.market_agreement_mrid = etree.SubElement(self.main_element, "marketAgreement.mRID")
        # self.market_agreement_time = etree.SubElement(self.main_element, "marketAgreement.createdDateTime")
        # self.activation_duration = etree.SubElement(self.main_element, "activation_ConstraintDuration.duration")
        # self.resting_duration = etree.SubElement(self.main_element, "resting_ConstraintDuration.duration")
        # self.minimum_duration = etree.SubElement(self.main_element, "minimum_ConstraintDuration.duration")
        # self.maximum_duration = etree.SubElement(self.main_element, "maximum_ConstraintDuration.duration")
        # self.standard_market_product = etree.SubElement(self.main_element, "standard_MarketProduct.marketProductType")
        # self.original_market_product = etree.SubElement(self.main_element, "original_MarketProduct.marketProductType")
        # self.validity_period = XMLPeriod(element_name="validity_Period.timeInterval")
        # self.inclusive_bids = etree.SubElement(self.main_element, "inclusiveBidsIdentification")
        # self.mkt_PSR_type = etree.SubElement(self.main_element, "mktPSRType.psrType")
        self.curve_type = etree.SubElement(self.main_element, "curveType")
        # self.original_market_doc_id = etree.SubElement(self.main_element, "original_MarketDocument.mRID")
        # self.original_market_doc_no = etree.SubElement(self.main_element, "original_MarketDocument.revisionNumber")
        # self.main_element.append(self.validity_period.main_element)
        self.add_period(points=self.points,
                        period_name='Period',
                        point_primary_field=self.point_primary_field,
                        point_secondary_field=self.point_secondary_field)

    @staticmethod
    def generate(series_start_time: str | datetime,
                 series_end_time: str | datetime,
                 business_type: str,
                 mrid: str,
                 resolution: str = None,
                 acquiring_domain: EICCodeOwner | dict = None,
                 connecting_domain: EICCodeOwner | dict = None,
                 measurement_unit: str = 'MAW',
                 sec_measurement_unit: str = 'P1',
                 flow_direction: str = None,
                 divisible: str = str(IndicatorType.YES.value),
                 curve_type: str = None,
                 auction_mrid: str = None,
                 values: DataToXMLPoint | list = None):
        """
        Generates ReserveBid time series

        :param series_start_time: 'Bid_TimeSeries.Period.timeInterval.start
        :param series_end_time: 'Bid_TimeSeries.Period.timeInterval.end'
        :param resolution: 'Bid_TimeSeries.Period.resolution'
        :param business_type: 'Bid_TimeSeries.businessType' (what is the time series about) See BusinessType
        :param mrid: 'Bid_TimeSeries.mRID' (id field)
        :param acquiring_domain:  'Bid_TimeSeries.acquiring_Domain.mRID' (where the product is delivered)
        :param connecting_domain: 'Bid_TimeSeries.connecting_Domain.mRID' (where the product is)
        :param measurement_unit: 'Bid_TimeSeries.quantity_Measurement_Unit.name' See MeasurementUnitType
        :param sec_measurement_unit: 'Bid_TimeSeries.price_Measurement_Unit.name' See MeasurementUnitType
        :param flow_direction: 'Bid_TimeSeries.flowDirection.direction' See FlowDirectionType
        :param divisible: 'Bid_TimeSeries.divisible' See IndicatorType
        :param curve_type: 'Bid_TimeSeries.curveType' See CurveType
        :param auction_mrid: 'Bid_TimeSeries.auction.mRID'
        :param values: 'Bid_TimeSeries.Period.Point.quantity.quantity'
        :return: ReserveBid timeseries
        """
        time_series = ReserveBidXMLTimeSeries(points=values)
        new_resolution = time_series.calculate_point_resolution(start_time=series_start_time,
                                                                end_time=series_end_time)
        series_start_time = convert_datetime_to_string_utc(series_start_time, output_format=XML_DATE_FORMAT)
        series_end_time = convert_datetime_to_string_utc(series_end_time, output_format=XML_DATE_FORMAT)
        time_series.mrid_element.text = str(mrid)
        time_series.business_type.text = str(business_type)
        if isinstance(acquiring_domain, dict):
            acquiring_domain = dict_to_dataclass(EICCodeOwner, acquiring_domain)
        if isinstance(connecting_domain, dict):
            connecting_domain = dict_to_dataclass(EICCodeOwner, connecting_domain)
        time_series.acquiring_domain.text = str(acquiring_domain.mRID)
        time_series.connecting_domain.text = str(connecting_domain.mRID)
        time_series.measurement_unit.text = str(measurement_unit)
        if time_series.price_measurement_unit is not None:
            time_series.price_measurement_unit.text = sec_measurement_unit
        time_series.divisible.text = str(divisible)
        time_series.flow_direction.text = str(flow_direction)
        time_series.curve_type.text = str(curve_type)
        time_series.period.time_series_start.text = str(series_start_time)
        time_series.period.time_series_end.text = str(series_end_time)
        if auction_mrid is not None:
            time_series.auction.text = str(auction_mrid[:60])
        resolution_value = time_delta_to_str(resolution) or new_resolution
        time_series.period.resolution.text = str(resolution_value)
        return time_series


@dataclass
class ReserveBidXMLDocument(XMLDocument):

    def __init__(self):
        """
        Initializes ReserveBid document header
        """
        super().__init__(document_name=RESERVE_BID_DOCUMENT_NAME,
                         period_name='reserveBid_Period.timeInterval',
                         root_value="urn:iec62325.351:tc57wg16:451-7:reservebiddocument:7:6")
        self.domain_mrid = etree.SubElement(self.main_element, "domain.mRID", codingScheme="A01")
        # self.subject_mrid = etree.SubElement(self.main_element, "subject_MarketParticipant.mRID", codingScheme="A01")
        # self.subject_role = etree.SubElement(self.main_element, "subject_MarketParticipant.marketRole.type")

    @staticmethod
    def generate(receiver: EICParty | dict,
                 sender: EICParty | dict,
                 doc_start_time: str | datetime,
                 doc_end_time: str | datetime,
                 created_at: str | datetime = None,
                 mrid: str = None,
                 process_type: str = None,
                 message_type: str = None,
                 revision_number: int = 1,
                 domain_mrid: str = None):
        """
        Generates ReserveBid document header

        :param receiver: 'ReserveBid_MarketDocument.receiver_MarketParticipant.mRID' and 'marketRole.type'
        :param sender:'ReserveBid_MarketDocument.sender_MarketParticipant.mRID' and 'marketRole.type'
        :param doc_start_time: 'ReserveBid_MarketDocument.reserveBid_Period.timeInterval.start'
        :param doc_end_time:'ReserveBid_MarketDocument.reserveBid_Period.timeInterval.end'
        :param created_at: 'ReserveBid_MarketDocument.createdDateTime'
        :param mrid: 'ReserveBid_MarketDocument.mRID'
        :param process_type: 'ReserveBid_MarketDocument.process.processType' See ProcessType
        :param message_type: 'ReserveBid_MarketDocument.type' See MessageType
        :param revision_number: 'ReserveBid_MarketDocument.revisionNumber'
        :param domain_mrid: 'ReserveBid_MarketDocument.domain.mRID'
        :return: XMLElementTree of document header, id
        """
        reserve_bid_document = ReserveBidXMLDocument()
        reserve_bid_document.add_parameters(receiver=receiver,
                                            sender=sender,
                                            doc_start_time=doc_start_time,
                                            doc_end_time=doc_end_time,
                                            created_at=created_at,
                                            mrid=mrid,
                                            process_type=process_type,
                                            message_type=message_type,
                                            revision_number=revision_number,
                                            # domain_mrid=domain_mrid
                                            )
        reserve_bid_document.domain_mrid.text = domain_mrid
        return reserve_bid_document, mrid
