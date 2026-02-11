from dataclasses import dataclass
import xml.dom.minidom
from datetime import datetime

import pytz

from lxml import etree

from py.common.functions import dict_to_dataclass
from py.common.time_functions import convert_datetime_to_string_utc
from py.common.ref_constants import CALC_DATE_KEY, PROCESS_TYPE_KEY, VERSION_KEY, MESSAGE_TYPE_KEY, SENDER_MRID_KEY, \
    SENDER_MARKET_ROLE_KEY, LFC_BLOCK_MRID_KEY
from py.data_classes.elastic.elastic_data_models import DataToXMLPoint
from py.data_classes.task_classes import EICParty, EICCodeOwner
from py.procurement.procurement_common import get_time_time_delta_for_point

XML_DATE_FORMAT = '%Y-%m-%dT%H:%M'

XML_TO_RESULT_MAP = {'createdDateTime': CALC_DATE_KEY,
                     'process.processType': PROCESS_TYPE_KEY,
                     'revisionNumber': VERSION_KEY,
                     'type': MESSAGE_TYPE_KEY,
                     'domain.mRID': LFC_BLOCK_MRID_KEY,
                     'sender_MarketParticipant.mRID': SENDER_MRID_KEY,
                     'sender_MarketParticipant.marketRole.type': SENDER_MARKET_ROLE_KEY}


def to_string(element: etree.Element):
    """
    Prints element to string

    :param element: input element tree element
    :return: None
    """
    xml_string = etree.tostring(element, encoding='utf-8')
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")
    pretty_xml_as_string = "\n".join(line for line in pretty_xml_as_string.split("\n") if line.strip())
    print(pretty_xml_as_string)


@dataclass
class XMLElement:
    def __init__(self, element_name: str, root_value: str = None):
        if root_value:
            self.main_element = etree.Element(element_name, xmlns=root_value)
        else:
            self.main_element = etree.Element(element_name)

    def to_string(self):
        return to_string(self.main_element)

    def to_xml(self):
        return self.main_element


@dataclass
class XMLPoint(XMLElement):
    def __init__(self,
                 element_name: str = 'Point',
                 position: int = 1,
                 primary_field: str = 'quantity',
                 primary_field_value = None,
                 secondary_field: str = 'secondaryQuantity',
                 secondary_field_value= None):
        super().__init__(element_name=element_name)
        self.position = etree.SubElement(self.main_element, "position")
        self.quantity = etree.SubElement(self.main_element, primary_field)
        self.secondaryQuantity = None
        if secondary_field_value is not None:
            self.secondaryQuantity = etree.SubElement(self.main_element, secondary_field)
        self.position.text = str(position)
        self.quantity.text = str(primary_field_value)
        if self.secondaryQuantity is not None:
            self.secondaryQuantity.text = str(secondary_field_value)



@dataclass
class XMLPeriod(XMLElement):
    def __init__(self, element_name: str = 'Period'):
        super().__init__(element_name=element_name)
        self.interval = etree.SubElement(self.main_element, "timeInterval")
        self.time_series_start = etree.SubElement(self.interval, "start")
        self.time_series_end = etree.SubElement(self.interval, "end")
        self.resolution = etree.SubElement(self.main_element, "resolution")

    def add_point(self, point: XMLPoint):
        self.main_element.append(point.main_element)

    def add_points(self,
                   points: DataToXMLPoint | list = None,
                   primary_field: str = 'quantity',
                   secondary_field: str = 'secondaryQuantity'):
        if not isinstance(points, list):
            points = [points]
        points = sorted(points, key=lambda x: x.position)
        for point in points:
            xml_point = XMLPoint(position=point.position,
                                 primary_field=primary_field,
                                 primary_field_value=point.primary_quantity,
                                 secondary_field=secondary_field,
                                 secondary_field_value= point.secondary_quantity)
            self.add_point(xml_point)


@dataclass
class XMLTimeSeries(XMLElement):
    def __init__(self,
                 points: DataToXMLPoint | list,
                 element_name: str = 'TimeSeries',
                 point_primary_field: str ='quantity',
                 point_secondary_field: str ='secondaryQuantity'):
        super().__init__(element_name=element_name)
        if isinstance(points, DataToXMLPoint):
            points = [points]
        self.points = points
        self.period = None
        self.point_primary_field = point_primary_field
        self.point_secondary_field = point_secondary_field

    def add_period(self,
                   points: DataToXMLPoint | list = None,
                   period_name: str = 'Period',
                   point_primary_field: str = 'quantity',
                   point_secondary_field: str = 'secondaryQuantity'):
        self.period = XMLPeriod(period_name)
        self.main_element.append(self.period.main_element)
        if isinstance(points, DataToXMLPoint):
            points = [points]
        if points:
            self.add_points(points=points,
                            primary_field=point_primary_field,
                            secondary_field=point_secondary_field)

    def add_point(self, point: XMLPoint):
        self.period.main_element.append(point.main_element)

    def add_points(self,
                   points: DataToXMLPoint | list = None,
                   primary_field: str = 'quantity',
                   secondary_field: str = 'secondaryQuantity'):
        points = sorted(points, key=lambda x: x.position)
        for point in points:
            xml_point = XMLPoint(position=point.position,
                                 primary_field=primary_field,
                                 primary_field_value=point.primary_quantity,
                                 secondary_field=secondary_field,
                                 secondary_field_value= point.secondary_quantity)
            self.add_point(xml_point)

    def calculate_point_resolution(self, start_time: str | datetime,
                                   end_time: str | datetime,
                                   points: int = None):
        points = points or len(self.points)
        new_resolution = get_time_time_delta_for_point(start_time=start_time,
                                                       end_time=end_time,
                                                       number_of_points=points)
        return new_resolution

    @staticmethod
    def generate(*args, **kwargs):
        pass

@dataclass
class XMLBidTimeSeries(XMLTimeSeries):
    def __init__(self,
                 points: DataToXMLPoint | list = None,
                 element_name: str = 'TimeSeries',
                 point_primary_field: str = 'quantity',
                 point_secondary_field: str = 'secondaryQuantity'):
        super().__init__(points=points,
                         element_name=element_name,
                         point_primary_field=point_primary_field,
                         point_secondary_field=point_secondary_field)


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
                 curve_type: str = None,
                 values: DataToXMLPoint | list = None):
        pass


@dataclass
class XMLATCTimeSeries(XMLTimeSeries):
    def __init__(self,
                 points: DataToXMLPoint | list = None,
                 element_name: str = 'TimeSeries',
                 point_primary_field: str = 'quantity',
                 point_secondary_field: str = 'secondaryQuantity'):
        super().__init__(points=points,
                         element_name=element_name,
                         point_primary_field=point_primary_field,
                         point_secondary_field=point_secondary_field)


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
                 values: DataToXMLPoint | list = None):
        pass


@dataclass
class XMLDocument(XMLElement):

    def __init__(self,
                 document_name: str,
                 root_value: str = None,
                 period_name: str = 'period.timeInterval',
                 inject_domain: bool = False):
        super().__init__(element_name=document_name, root_value=root_value)
        self.doc_mrid = etree.SubElement(self.main_element, "mRID")
        self.revision_number = etree.SubElement(self.main_element, "revisionNumber")
        self.doc_type = etree.SubElement(self.main_element, "type")
        self.process_type = etree.SubElement(self.main_element, "process.processType")

        self.sender_mrid = etree.SubElement(self.main_element, "sender_MarketParticipant.mRID", codingScheme="A01")
        self.sender_role = etree.SubElement(self.main_element, "sender_MarketParticipant.marketRole.type")
        self.receiver_mrid = etree.SubElement(self.main_element, "receiver_MarketParticipant.mRID", codingScheme="A01")
        self.receiver_role = etree.SubElement(self.main_element, "receiver_MarketParticipant.marketRole.type")
        self.created_datetime = etree.SubElement(self.main_element, "createdDateTime")
        if inject_domain:
            self.domain_mrid = etree.SubElement(self.main_element, "area_Domain.mRID", codingScheme="A01")
        self.period = etree.SubElement(self.main_element, period_name)
        self.period_start = etree.SubElement(self.period, "start")
        self.period_end = etree.SubElement(self.period, "end")
        # self.domain_mrid = etree.SubElement(self.main_element, "domain.mRID", codingScheme="A01")


    def to_string(self) -> str:
        xml_string = etree.tostring(self.main_element, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml_as_string = dom.toprettyxml(indent="  ")
        pretty_xml_as_string = "\n".join(line for line in pretty_xml_as_string.split("\n") if line.strip())
        return pretty_xml_as_string

    def add_timeseries(self, time_series: XMLTimeSeries):
        self.main_element.append(time_series.main_element)

    def to_xml(self):
        """
        Converts document to xml
        :return:
        """
        # Retrieve document's mRID for file naming
        # random_string = ''.join(choices(ascii_letters, k=6))
        # mrid = self.main_element.findtext('mRID')[:-6] + random_string  # 2 receivers -> 2 unique file names
        # Did not see benefit of having time horizon but just in case it's a matter of taste, kept process type
        # process_type = self.main_element.findtext("process.processType")
        # Prepares xml string for output
        xml_string = etree.tostring(self.main_element, encoding='utf-8')
        # Converts single line xml to well formatted xml + ads xml declaration
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml_as_string = dom.toprettyxml(indent="  ", encoding='utf-8')
        return pretty_xml_as_string.decode('utf-8')


    def add_parameters(self,
                       receiver: EICParty | dict,
                       sender: EICParty | dict,
                       doc_start_time: str | datetime,
                       doc_end_time: str | datetime,
                       created_at: str | datetime = None,
                       mrid: str = None,
                       process_type: str = None,
                       message_type: str = None,
                       revision_number: int = 1,
                       domain_mrid: str = None,
                       ):
        if isinstance(receiver, dict):
            receiver = dict_to_dataclass(EICParty, receiver)
        if isinstance(sender, dict):
            sender = dict_to_dataclass(EICParty, sender)
        doc_start_time = convert_datetime_to_string_utc(doc_start_time, output_format=XML_DATE_FORMAT)
        doc_end_time = convert_datetime_to_string_utc(doc_end_time, output_format=XML_DATE_FORMAT)
        if not created_at:
            created_at = datetime.now(pytz.utc)
        created_at = convert_datetime_to_string_utc(created_at, output_format='%Y-%m-%dT%H:%M:%S')
        self.doc_mrid.text = mrid
        self.process_type.text = process_type
        self.doc_type.text = message_type
        self.revision_number.text = str(revision_number)
        self.sender_mrid.text = sender.mRID
        self.sender_role.text = sender.market_role_type.value
        self.receiver_mrid.text = receiver.mRID
        self.receiver_role.text = receiver.market_role_type.value
        self.created_datetime.text = created_at
        self.period_start.text = doc_start_time
        self.period_end.text = doc_end_time
        # self.domain_mrid.text = domain_mrid
        if hasattr(self, 'domain_mrid'):
            self.domain_mrid.text = domain_mrid

    @staticmethod
    def generate(receiver: EICCodeOwner | dict,
                 sender: EICCodeOwner | dict,
                 doc_start_time: str | datetime,
                 doc_end_time: str | datetime,
                 created_at: str | datetime = None,
                 mrid: str = None,
                 process_type: str = None,
                 message_type: str = None,
                 revision_number: int = 1,
                 domain_mrid: str = None):
        pass
