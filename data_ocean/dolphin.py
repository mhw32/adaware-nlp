# -*- encoding: UTF-8 -*-

from __future__ import print_function
import httplib2
import os

# pip install --upgrade google-api-python-client
from oauth2client.file import Storage
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow

from apiclient import discovery, errors
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

    def _get_credentials(self):
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
            flow = OAuth2WebServerFlow(client_id=self.client_id,
                                       client_secret=self.client_secret,
                                       scope=self.oauth_scope,
                                       redirect_uri=self.redirect_uri)
            credentials = tools.run_flow(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials

    def connect(self):
        credentials = self._get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('drive', 'v3', http=http)

        self.http = http
        self.service = service

    def _search_folder(self, parent, recursive=False, name='root', tree={}):
        if self.service is None:
            raise ValueError('please connect() first.')

        files = self.service.files().list(q= "'{}' in parents and trashed=false".format(parent)).execute()
        tasks = files.get('files', [])

        if recursive:
            for f in tasks:
                if f['mimeType'] == 'application/vnd.google-apps.folder':
                    tasks = self._search_folder(f['id'],
                                                recursive=recursive,
                                                name=f['name'],
                                                tree=tree)
                    tree[f['name']] = tasks
                else:
                    if name in tree:
                        tree[name].append(f)
                    else:
                        tree[name] = [f]

        return tree

    def _download_file(self, drive_file, write_file):
        """ Download a file's content.

            Args
            ----
            service : Drive API service instance.
            drive_file : Drive File instance.

            Returns
            -------
            File's content if successful, None otherwise.
        """
        download_url = drive_file.get('downloadUrl')
        if download_url:
            resp, content = service._http.request(download_url)
            if resp.status == 200:
                print('Status: {}'.format(resp))
                with open(write_file, 'w') as fw:
                    fw.write(content)
                return
            else:
                print('An error occurred: {}'.format(resp))
                return None
        else:
            print('The file doesn\'t have any content stored on Drive.')
            return None

client_id = os.environ['ADA_GOOGLE_CLIENT_ID']
client_secret = os.environ['ADA_GOOGLE_CLIENT_SECRET']
credentials_file = os.path.abspath('_passwords/Adaware-054594a036d1.json')

dolphin = Dolphin(client_id, client_secret, credentials_file)
dolphin.connect()
print(dolphin._search_folder('0B6JRxhFLmKU0ak4xSUwwYnZld2c', recursive=True))
