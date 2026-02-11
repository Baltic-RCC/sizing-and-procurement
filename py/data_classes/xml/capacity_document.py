from dataclasses import dataclass
from datetime import datetime

from lxml import etree

from py.common.functions import dict_to_dataclass
from py.common.time_functions import convert_datetime_to_string_utc, time_delta_to_str
from py.data_classes.elastic.elastic_data_models import DataToXMLPoint
from py.data_classes.task_classes import EICCodeOwner, EICParty
from py.data_classes.xml.base_xml import XMLDocument, XML_DATE_FORMAT, XMLATCTimeSeries, XML_TO_RESULT_MAP
from py.common.ref_constants import POINT_RESOLUTION_KEY, AVAILABLE_POSITION_KEY, AVAILABLE_VALUE_KEY, \
    PERCENTAGE_VALUE_KEY, MRID_KEY, LFC_BLOCK_MRID_KEY, BUSINESS_TYPE_KEY, IN_DOMAIN_MRID_KEY, OUT_DOMAIN_MRID_KEY, \
    CURVE_TYPE_KEY, PERCENTAGE_UNIT_KEY, AVAILABLE_UNIT_KEY, SERIES_VALID_TO_KEY, SERIES_VALID_FROM_KEY, VALID_FROM_KEY, \
    VALID_TO_KEY, PRODUCT_KEY

CAPACITY_DOCUMENT_NAME = 'Capacity_MarketDocument'
CAPACITY_XML_TO_RESULT_MAP = {'Period.resolution': POINT_RESOLUTION_KEY,
                              'Point.position': AVAILABLE_POSITION_KEY,
                              'Point.quantity': AVAILABLE_VALUE_KEY,
                              'Point.secondaryQuantity': PERCENTAGE_VALUE_KEY,
                              'TimeSeries.businessType': BUSINESS_TYPE_KEY,
                              'TimeSeries.curveType': CURVE_TYPE_KEY,
                              'TimeSeries.mRID': MRID_KEY,
                              'TimeSeries.in_Domain.mRID': IN_DOMAIN_MRID_KEY,
                              'TimeSeries.measurement_Unit.name': AVAILABLE_UNIT_KEY,
                              'TimeSeries.out_Domain.mRID': OUT_DOMAIN_MRID_KEY,
                              'TimeSeries.product': PRODUCT_KEY,
                              'TimeSeries.secondary_Measurement_Unit.name': PERCENTAGE_UNIT_KEY,
                              'domain.mRID': LFC_BLOCK_MRID_KEY,
                              'Period.timeInterval.end': SERIES_VALID_TO_KEY,
                              'Period.timeInterval.start': SERIES_VALID_FROM_KEY,
                              'utc_start': VALID_FROM_KEY,
                              'utc_end': VALID_TO_KEY}

CAPACITY_XML_TO_RESULT_MAP = {**XML_TO_RESULT_MAP, **CAPACITY_XML_TO_RESULT_MAP}


@dataclass
class CapacityXMLTimeSeries(XMLATCTimeSeries):

    def __init__(self, points: DataToXMLPoint | list = 1):
        """
        Initializes Capacity Timeseries

        :param points: how many points to be added
        """
        super().__init__(points=points,
                         element_name='TimeSeries',
                         point_primary_field='quantity',
                         point_secondary_field='secondaryQuantity')
        self.mrid_element = etree.SubElement(self.main_element, "mRID")
        self.business_type = etree.SubElement(self.main_element, "businessType")
        self.product = etree.SubElement(self.main_element, "product")
        self.in_domain = etree.SubElement(self.main_element, "in_Domain.mRID", codingScheme="A01")
        self.out_domain = etree.SubElement(self.main_element, "out_Domain.mRID", codingScheme="A01")
        self.measurement_unit = etree.SubElement(self.main_element, "measurement_Unit.name")
        self.secondary_measurement_unit = None
        if any([k.secondary_quantity for k in self.points]):
            self.secondary_measurement_unit = etree.SubElement(self.main_element, "secondary_Measurement_Unit.name")
        self.auction = etree.SubElement(self.main_element, "auction.mRID")
        # self.auction_category = etree.SubElement(self.main_element, "auction.category")
        self.curve_type = etree.SubElement(self.main_element, "curveType")
        # self.connecting_line = etree.SubElement(self.main_element, "connectingLine_RegisteredResource.mRID")
        # self.market_participant_mrid = etree.SubElement(self.main_element, "requesting_MarketParticipant.mRID")
        # self.market_participant_type = etree.SubElement(self.main_element, "requesting_MarketParticipant.marketRole.type")
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
                 in_domain: EICCodeOwner | dict = None,
                 out_domain: EICCodeOwner | dict = None,
                 measurement_unit: str = 'MAW',
                 sec_measurement_unit: str = 'percentage',
                 curve_type: str = None,
                 product: str = None,
                 auction_mrid: str = None,
                 values: DataToXMLPoint | list = None):
        """
        Generates Capacity time series

        :param series_start_time: 'TimeSeries.Period.timeInterval.start
        :param series_end_time: 'TimeSeries.Period.timeInterval.end'
        :param resolution: 'TimeSeries.Period.resolution'
        :param business_type: 'TimeSeries.businessType' (what is the time series about) See BusinessType
        :param mrid: 'TimeSeries.mRID' (id field)
        :param in_domain:  'TimeSeries.in_Domain.mRID' (where the product is delivered)
        :param out_domain: 'TimeSeries.out_Domain.mRID' (where the product is)
        :param measurement_unit: 'TimeSeries.measurement_Unit.name' See MeasurementUnitType
        :param sec_measurement_unit: 'TimeSeries.secondary_Measurement_Unit.name' See MeasurementUnitType
        :param product: 'TimeSeries.product.direction' See ProductType
        :param curve_type: 'TimeSeries.curveType' See CurveType
        :param auction_mrid: 'TimeSeries.auction.mRID'
        :param values: 'TimeSeries.Period.Point.quantity'
        :return: ReserveBid timeseries
        """
        time_series = CapacityXMLTimeSeries(points=values)
        new_resolution = time_series.calculate_point_resolution(start_time=series_start_time,
                                                                end_time=series_end_time)
        series_start_time = convert_datetime_to_string_utc(series_start_time, output_format=XML_DATE_FORMAT)
        series_end_time = convert_datetime_to_string_utc(series_end_time, output_format=XML_DATE_FORMAT)
        time_series.mrid_element.text = str(mrid)
        time_series.business_type.text = str(business_type)
        if isinstance(in_domain, dict):
            in_domain = dict_to_dataclass(EICCodeOwner, in_domain)
        if isinstance(out_domain, dict):
            out_domain = dict_to_dataclass(EICCodeOwner, out_domain)
        time_series.in_domain.text = str(in_domain.mRID)
        time_series.out_domain.text = str(out_domain.mRID)
        time_series.measurement_unit.text = str(measurement_unit)
        if time_series.secondary_measurement_unit is not None:
            time_series.secondary_measurement_unit.text = str(sec_measurement_unit)
        time_series.curve_type.text = str(curve_type)
        time_series.product.text = str(product)
        time_series.period.time_series_start.text = str(series_start_time)
        time_series.period.time_series_end.text = str(series_end_time)
        if auction_mrid is not None:
            time_series.auction.text = str(auction_mrid[:60])
        resolution_value = time_delta_to_str(resolution) or new_resolution
        time_series.period.resolution.text = str(resolution_value)
        return time_series


@dataclass
class CapacityXMLDocument(XMLDocument):

    def __init__(self):
        """
        Initializes Capacity document header
        """
        super().__init__(document_name=CAPACITY_DOCUMENT_NAME,
                         root_value="urn:iec62325.351:tc57wg16:451-3:capacitydocument:8:3")
        # self.__doc_status = etree.SubElement(self.main_element, "docStatus")
        # self.status_value = etree.SubElement(self.__doc_status, "value")
        # self.received_market_document_mrid = etree.SubElement(self.main_element, "received_MarketDocument.mRID")
        # self.received_market_document_no = etree.SubElement(self.main_element, "received_MarketDocument.revisionNumber")
        self.domain_mrid = etree.SubElement(self.main_element, "domain.mRID", codingScheme="A01")

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
        :param receiver: 'Capacity_MarketDocument.receiver_MarketParticipant.mRID' and 'marketRole.type'
        :param sender:'Capacity_MarketDocument.sender_MarketParticipant.mRID' and 'marketRole.type'
        :param doc_start_time: 'Capacity_MarketDocument.period.timeInterval.start'
        :param doc_end_time:'Capacity_MarketDocument.period.timeInterval.end'
        :param created_at: 'Capacity_MarketDocument.createdDateTime'
        :param mrid: 'Capacity_MarketDocument.mRID'
        :param process_type: 'Capacity_MarketDocument.process.processType' See ProcessType
        :param message_type: 'Capacity_MarketDocument.type' See MessageType
        :param revision_number: 'Capacity_MarketDocument.revisionNumber'
        :param domain_mrid: 'Capacity_MarketDocument.domain.mRID'
        :return: XMLElementTree of document header, id
        """
        capacity_document = CapacityXMLDocument()
        capacity_document.add_parameters(receiver=receiver,
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
        capacity_document.domain_mrid.text = domain_mrid
        return capacity_document, mrid
