# -*- encoding: UTF-8 -*-

from __future__ import print_function
import httplib2
import os

# pip install --upgrade google-api-python-client
from oauth2client.file import Storage
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

class Dolphin(object):
    def __init__(self,
                 client_id,
                 client_secret,
                 application_name):
        ''' A manager object to download datasets for the user. All data is currently
            stored in the private Ada Google Drive.

            Args
            ----
            client_id : string
                        Google API client id
            client_secret : string
                            Google API client secret
            application_name : string
                               name of the app
        '''

        self.client_id = client_id
        self.client_secret = client_secret
        self.application_name = application_name

        oauth_scope = 'https://www.googleapis.com/auth/drive'
        redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
        self.oauth_scope = oauth_scope
        self.redirect_uri = redirect_uri
        self.service = None

    def get_credentials():
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir,
                                       'ada_gdrive_credentials.json')

        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(self.client_secret, self.oauth_scope)
            flow.user_agent = self.application_name
            if flags:
                credentials = tools.run_flow(flow, store, flags)
            else: # Needed only for compatibility with Python 2.6
                credentials = tools.run(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials

    def connect(self):
        credentials = get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)

        self.http = http
        self.service = service

    def list_files(self):
        if self.service is None:
            raise ValueError('please connect() first.')

        results = service.files().list(
                pageSize=10,fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print('{0} ({1})'.format(item['name'], item['id']))

client_id = os.environ['ADA_GOOGLE_CLIENT_ID']
client_secret = os.environ['ADA_GOOGLE_CLIENT_SECRET']
credentials_file = os.path.abspath('_passwords/Adaware-054594a036d1.json')

dolphin = Dolphin(client_id, client_secret, credentials_file)
dolphin.connect()
dolphin.list_files()
