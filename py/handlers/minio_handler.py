from datetime import timedelta

import pytz
import urllib3
import logging
import config
from io import BytesIO
from py.common.config_parser import parse_app_properties

logger = logging.getLogger(__name__)

urllib3.disable_warnings()

try:
    from brcc_apis.minio_api import ObjectStorage
except ModuleNotFoundError:

    import requests
    from lxml import etree
    from minio import Minio, error
    import sys
    import mimetypes
    import re
    from datetime import datetime
    from aniso8601 import parse_duration, parse_datetime


    class ObjectStorage:

        def __init__(self, server: str, username: str, password: str):
            self.server = server
            self.username = username
            self.password = password
            self.token_expiration = datetime.now(pytz.utc)
            self.http_client = urllib3.PoolManager(
                maxsize=20,
                cert_reqs='CERT_NONE',
                # cert_reqs='CERT_REQUIRED',
                # ca_certs='/usr/local/share/ca-certificates/CA-Bundle.crt'
            )

            # Init client
            self.__create_client()

        def __create_client(self):

            if self.token_expiration < (datetime.utcnow() + parse_duration("PT1M")):
                credentials = self.__get_credentials()

                self.token_expiration = parse_datetime(credentials['Expiration']).replace(tzinfo=None)
                self.client = Minio(endpoint=self.server,
                                    access_key=credentials['AccessKeyId'],
                                    secret_key=credentials['SecretAccessKey'],
                                    session_token=credentials['SessionToken'],
                                    secure=True,
                                    http_client=self.http_client,
                                    )
                logger.info(f"Connection created to Minio at {self.server} as {self.username}")

        def __get_credentials(self, action: str = "AssumeRoleWithLDAPIdentity", version: str = "2011-06-15"):
            """
            Method to get temporary credentials for LDAP user

            :param action: string of action
            :param version: version
            :return:
            """
            # Define LDAP service user parameters
            params = {
                "Action": action,
                "LDAPUsername": self.username,
                "LDAPPassword": self.password,
                "Version": version,
            }

            # Sending request for temporary credentials and parsing it out from returned xml
            response = requests.post(f"https://{self.server}", params=params, verify=False).content
            credentials = {}
            root = etree.fromstring(response)
            et = root.find("{*}AssumeRoleWithLDAPIdentityResult/{*}Credentials")
            for element in et:
                _, _, tag = element.tag.rpartition("}")
                credentials[tag] = element.text

            return credentials

        def upload_object(self,
                          file_path_or_file_object: str | BytesIO,
                          bucket_name: str,
                          metadata: dict | None = None):
            """
            Method to upload file to Minio storage

            :param file_path_or_file_object: file path or BytesIO object
            :param bucket_name: bucket name
            :param metadata: object metadata
            :return: response from Minio
            """
            file_object = file_path_or_file_object

            if isinstance(file_path_or_file_object, str):
                file_object = open(file_path_or_file_object, "rb")
                length = sys.getsizeof(file_object)
            else:
                length = file_object.getbuffer().nbytes

            # Just to be sure that pointer is at the beginning of the content
            file_object.seek(0)

            response = self.client.put_object(
                bucket_name=bucket_name,
                object_name=file_object.name,
                data=file_object,
                length=length,
                content_type=mimetypes.guess_type(file_object.name)[0],
                metadata=metadata
            )

            return response


parse_app_properties(globals(), config.paths.config.minio)
DAYS_TO_STORE_DATA_IN_MINIO = 7  # Max allowed by Minio

# Constants to be loaded in from config file
PY_MINIO_SERVER = MINIO_SERVER
PY_MINIO_USERNAME = MINIO_USERNAME
PY_MINIO_PASSWORD = MINIO_PASSWORD
PY_MINIO_BUCKET = MINIO_BUCKET
PY_MINIO_PATH = MINIO_PATH_PROCUREMENT


class ReportMinioStorage(ObjectStorage):

    def __init__(self, server=PY_MINIO_SERVER, username=PY_MINIO_USERNAME, password=PY_MINIO_PASSWORD):
        super().__init__(server, username, password)

    def save_file_to_minio_with_link(self,
                                     buffer='',
                                     file_name: str = None,
                                     minio_bucket: str = PY_MINIO_BUCKET,
                                     minio_path: str = PY_MINIO_PATH):
        """
        Posts log as a file to minio

        :param buffer: logs as a string
        :param file_name: if given
        :param minio_bucket:
        :param minio_path:
        :return: file name and link to file, the link to the file
        """
        link_to_file = None
        if buffer != '' and buffer is not None:
            # check if the given bucket exists
            if not self.client.bucket_exists(bucket_name=minio_bucket):
                logger.warning(f"{minio_bucket} does not exist")
                return link_to_file
            if not isinstance(buffer, BytesIO):
                file_object = BytesIO(str.encode(buffer))
            else:
                file_object = buffer
                file_name = file_name or buffer.name
            if minio_path:
                extension = file_name.split(".")[1].upper()
                file_name = minio_path + '/' + extension + '/' + file_name.removeprefix('/')
            file_object.name = file_name
            self.upload_object(file_path_or_file_object=file_object, bucket_name=minio_bucket)
            time_to_expire = timedelta(days=DAYS_TO_STORE_DATA_IN_MINIO)
            link_to_file = self.client.get_presigned_url(method="GET",
                                                         bucket_name=minio_bucket,
                                                         object_name=file_object.name,
                                                         expires=time_to_expire)
        return file_name, link_to_file


def save_file_to_minio_with_link(file_to_save,
                                 file_name: str = None,
                                 minio_bucket: str = PY_MINIO_BUCKET,
                                 minio_path: str = PY_MINIO_PATH,
                                 minio_instance: ReportMinioStorage = None
                                 ):
    """
    Saves file to minio, prints link to console

    :param minio_instance: minio instance if specified
    :param minio_path: folder path in minio
    :param minio_bucket: bucket in minio
    :param file_name: file name if specified
    :param file_to_save: file instance to be saved
    :return: link to file in minio
    """
    minio_instance = minio_instance or ReportMinioStorage()
    link_value = minio_instance.save_file_to_minio_with_link(buffer=file_to_save,
                                                             file_name=file_name,
                                                             minio_bucket=minio_bucket,
                                                             minio_path=minio_path
                                                             )
    logger.info(f"Report can be downloaded at {link_value}")
    return link_value
