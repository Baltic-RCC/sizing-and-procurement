from dataclasses import dataclass
from datetime import datetime

from lxml import etree

from py.common.functions import dict_to_dataclass
from py.common.time_functions import convert_datetime_to_string_utc, time_delta_to_str
from py.data_classes.elastic.elastic_data_models import DataToXMLPoint
from py.data_classes.task_classes import EICCodeOwner, EICParty
from py.data_classes.xml.base_xml import XMLDocument, XML_DATE_FORMAT, XMLBidTimeSeries, XML_TO_RESULT_MAP
from py.common.ref_constants import POINT_RESOLUTION_KEY, AVAILABLE_POSITION_KEY, AVAILABLE_VALUE_KEY, \
    PERCENTAGE_VALUE_KEY, MRID_KEY, LFC_BLOCK_MRID_KEY, BUSINESS_TYPE_KEY, DOMAIN_MRID_KEY, CURVE_TYPE_KEY, \
    DIRECTION_CODE_KEY, PERCENTAGE_UNIT_KEY, AVAILABLE_UNIT_KEY, SERIES_VALID_TO_KEY, SERIES_VALID_FROM_KEY, \
    VALID_FROM_KEY, VALID_TO_KEY

BALANCING_XML_DOCUMENT_NAME = 'Balancing_MarketDocument'
BALANCING_XML_TO_RESULT_MAP = {'Period.resolution': POINT_RESOLUTION_KEY,
                               'Point.position': AVAILABLE_POSITION_KEY,
                               'Point.quantity': AVAILABLE_VALUE_KEY,
                               'Point.secondaryQuantity': PERCENTAGE_VALUE_KEY,
                               'TimeSeries.mRID': MRID_KEY,
                               'TimeSeries.acquiring_Domain.mRID': LFC_BLOCK_MRID_KEY,
                               'TimeSeries.businessType': BUSINESS_TYPE_KEY,
                               'TimeSeries.connecting_Domain.mRID': DOMAIN_MRID_KEY,
                               'TimeSeries.curveType': CURVE_TYPE_KEY,
                               'TimeSeries.flowDirection.direction': DIRECTION_CODE_KEY,
                               'TimeSeries.price_Measurement_Unit.name': PERCENTAGE_UNIT_KEY,
                               'TimeSeries.quantity_Measurement_Unit.name': AVAILABLE_UNIT_KEY,
                               'area_Domain.mRID': LFC_BLOCK_MRID_KEY,
                               'Period.timeInterval.end': SERIES_VALID_TO_KEY,
                               'Period.timeInterval.start': SERIES_VALID_FROM_KEY,
                               'utc_start': VALID_FROM_KEY,
                               'utc_end': VALID_TO_KEY}

BALANCING_XML_TO_RESULT_MAP = {**XML_TO_RESULT_MAP, **BALANCING_XML_TO_RESULT_MAP}


@dataclass
class BalancingXMLTimeSeries(XMLBidTimeSeries):

    def __init__(self, points: DataToXMLPoint | list = None):
        """
        Initializes Balancing Timeseries

        :param points: how many points to be added
        """
        super().__init__(points=points,
                         element_name='TimeSeries',
                         point_primary_field='quantity',
                         point_secondary_field='secondaryQuantity')
        self.mrid_element = etree.SubElement(self.main_element, "mRID")
        self.business_type = etree.SubElement(self.main_element, "businessType")
        self.acquiring_domain = etree.SubElement(self.main_element, "acquiring_Domain.mRID", codingScheme="A01")
        self.connecting_domain = etree.SubElement(self.main_element, "connecting_Domain.mRID", codingScheme="A01")
        # self.market_agreement_type = etree.SubElement(self.main_element, "type_MarketAgreement.type")
        # self.standard_market_product = etree.SubElement(self.main_element, "standard_MarketProduct.marketProductType")
        # self.original_market_product = etree.SubElement(self.main_element, "original_MarketProduct.marketProductType")
        # self.mkt_PSR_type = etree.SubElement(self.main_element, "mktPSRType.psrType")
        self.flow_direction = etree.SubElement(self.main_element, "flowDirection.direction")
        # self.currency_unit = etree.SubElement(self.main_element, "currency_Unit.name")
        self.measurement_unit = etree.SubElement(self.main_element, "quantity_Measurement_Unit.name")
        self.price_measurement_unit = None
        if any([k.secondary_quantity for k in self.points]):
            self.price_measurement_unit = etree.SubElement(self.main_element, "price_Measurement_Unit.name")
        self.curve_type = etree.SubElement(self.main_element, "curveType")
        # self.cancelled_ts = etree.SubElement(self.main_element, "cancelledTS")
        self.auction = etree.SubElement(self.main_element, "auction.mRID")
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
                 divisible: str = None,
                 curve_type: str = None,
                 auction_mrid: str = None,
                 values: DataToXMLPoint | list = None):
        """
        Generates Balancing time series

        :param divisible: empty
        :param series_start_time: 'TimeSeries.Period.timeInterval.start
        :param series_end_time: 'TimeSeries.Period.timeInterval.end'
        :param resolution: 'TimeSeries.Period.resolution'
        :param business_type: 'TimeSeries.businessType' (what is the time series about) See BusinessType
        :param mrid: 'TimeSeries.mRID' (id field)
        :param acquiring_domain:  'TimeSeries.acquiring_Domain.mRID' (where the product is delivered)
        :param connecting_domain: 'TimeSeries.connecting_Domain.mRID' (where the product is)
        :param measurement_unit: 'TimeSeries.quantity_Measurement_Unit.name' See MeasurementUnitType
        :param sec_measurement_unit: 'TimeSeries.price_Measurement_Unit.name' See MeasurementUnitType
        :param flow_direction: 'TimeSeries.flowDirection.direction' See FlowDirectionType
        :param curve_type: 'TimeSeries.curveType' See CurveType
        :param auction_mrid: 'TimeSeries.auction.mRID'
        :param values: 'TimeSeries.Period.Point.quantity.quantity'
        :return: ReserveBid timeseries
        """
        time_series = BalancingXMLTimeSeries(points=values)
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
            time_series.price_measurement_unit.text = str(sec_measurement_unit)
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
class BalancingXMLDocument(XMLDocument):
    """
    Initializes Balancing document header
    """

    def __init__(self):
        super().__init__(document_name=BALANCING_XML_DOCUMENT_NAME,
                         root_value="urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:5",
                         inject_domain=True)
        # self.root = etree.Element("Balancing_MarketDocument")
        # self.domain_mrid = etree.SubElement(self.main_element, "area_Domain.mRID", codingScheme="A01")

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
                 domain_mrid: str = None
                 ):
        """
        Generates Balancing document header

        :param receiver: 'Balancing_MarketDocument.receiver_MarketParticipant.mRID' and 'marketRole.type'
        :param sender:'Balancing_MarketDocument.sender_MarketParticipant.mRID' and 'marketRole.type'
        :param doc_start_time: 'Balancing_MarketDocument.period.timeInterval.start'
        :param doc_end_time:'Balancing_MarketDocument.period.timeInterval.end'
        :param created_at: 'Balancing_MarketDocument.createdDateTime'
        :param mrid: 'Balancing_MarketDocument.mRID'
        :param process_type: 'Balancing_MarketDocument.process.processType' See ProcessType
        :param message_type: 'Balancing_MarketDocument.type' See MessageType
        :param revision_number: 'Balancing_MarketDocument.revisionNumber'
        :param domain_mrid: 'Balancing_MarketDocument.area_domain.mRID'
        :return: XMLElementTree of document header, id
        """
        balancing_document = BalancingXMLDocument()
        balancing_document.add_parameters(receiver=receiver,
                                          sender=sender,
                                          doc_start_time=doc_start_time,
                                          doc_end_time=doc_end_time,
                                          created_at=created_at,
                                          mrid=mrid,
                                          process_type=process_type,
                                          message_type=message_type,
                                          revision_number=revision_number)
        balancing_document.domain_mrid.text = domain_mrid
        return balancing_document, mrid
