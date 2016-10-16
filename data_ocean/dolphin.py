# -*- encoding: UTF-8 -*-

import os
import httplib2

# pip install --upgrade google-api-python-client
from oauth2client.file import Storage
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow


class Dolphin(object):
    def __init__(self, client_id, client_secret, credentials_file):
        ''' A manager object to download datasets for the user. All data is currently
            stored in the private Ada Google Drive.

            Args
            ----
            client_id : string
                        Google API client id
            client_secret : string
                            Google API client secret
            credentials_file : string
                               Path to Google credentials JSON file
        '''

        self.client_id = client_id
        self.client_secret = client_secret
        self.credentials_file = credentials_file

        oauth_scope = 'https://www.googleapis.com/auth/drive'
        redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
        self.oauth_scope = oauth_scope
        self.redirect_uri = redirect_uri
        self.service = None

    def connect(self):
        storage = Storage(self.credentials_file)
        credentials = storage.get()

        if credentials is None:
            # Run through the OAuth flow and retrieve credentials
            flow = OAuth2WebServerFlow(self.client_id,
                                       self.client_secret,
                                       self.oauth_scope,
                                       self.redirect_uri)
            authorize_url = flow.step1_get_authorize_url()
            print 'Go to the following link in your browser: ' + authorize_url
            code = raw_input('Enter verification code: ').strip()
            credentials = flow.step2_exchange(code)
            storage.put(credentials)

        # Create an httplib2.Http object and authorize it with our credentials
        http = httplib2.Http()
        http = credentials.authorize(http)
        drive_service = build('drive', 'v2', http=http)

        self.http = http
        self.service = drive_service

    def list_files(self):
        if self.service is None:
            raise ValueError('please connect() first.')

        page_token = None
        while True:
            param = {}
            if page_token:
                param['pageToken'] = page_token
            files = self.service.files().list(**param).execute()
            for item in files['items']:
                yield item
            page_token = files.get('nextPageToken')
            if not page_token:
                break

    def download_files(self):
        if self.service is None:
            raise ValueError('please connect() first.')

        for item in list_files(self.service):
            if item.get('title').upper().startswith('OFFER'):
                outfile = os.path.join(OUT_PATH, '%s.pdf' % item['title'])
                download_url = None
                if 'exportLinks' in item and 'application/pdf' in item['exportLinks']:
                    download_url = item['exportLinks']['application/pdf']
                elif 'downloadUrl' in item:
                    download_url = item['downloadUrl']
                else:
                    print 'ERROR getting %s' % item.get('title')
                    print item
                    print dir(item)
                if download_url:
                    print "downloading %s" % item.get('title')
                    resp, content = self.service._http.request(download_url)
                    if resp.status == 200:
                        if os.path.isfile(outfile):
                            print "ERROR, %s already exist" % outfile
                        else:
                            with open(outfile, 'wb') as f:
                                f.write(content)
                            print "OK"
                    else:
                        print 'ERROR downloading %s' % item.get('title')


client_id = os.environ['ADA_GOOGLE_CLIENT_ID']
client_secret = os.environ['ADA_GOOGLE_CLIENT_SECRET']
credentials_file = os.path.abspath('_passwords/Adaware-054594a036d1.json')

dolphin = Dolphin(client_id, client_secret, credentials_file)
dolphin.connect()
dolphin.list_files()

