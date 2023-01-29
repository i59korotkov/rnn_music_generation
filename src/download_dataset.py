import os
import urllib.request
import zipfile


URL = 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip'
ZIP_FILE_NAME = 'maestro-v3.0.0-midi.zip'
DATA_FOLDER = 'data/'

print('Downloading...')
urllib.request.urlretrieve(URL, DATA_FOLDER + ZIP_FILE_NAME)

print('Unzipping...')
with zipfile.ZipFile(DATA_FOLDER + ZIP_FILE_NAME, 'r') as zip_ref:
    zip_ref.extractall(DATA_FOLDER)
os.remove(DATA_FOLDER + ZIP_FILE_NAME)

print('Done.')
