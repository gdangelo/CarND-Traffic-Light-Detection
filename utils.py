import os
import urllib.request
import tarfile

def download_and_extract_model(model_name):
    if model_name:
        # Set complete URL to target for downloading model
        base = 'http://download.tensorflow.org/models/object_detection/'
        file = model_name + '.tar.gz'
        url = base + file

        # Download model
        if not os.path.exists(file):
            print('Downloading model {0}...'.format(model_name))
            url_opener = urllib.request.URLopener()
            url_opener.retrieve(url, file)
        else:
            print('Model already downloaded.')

        # Extract tar file
        if not os.path.exists(model_name):
            print('Extracting model archive {0}...'.format(file))
            tar_file = tarfile.open(file, 'r:gz')
            tar_file.extractall()
        else:
            print('Model archive already extracted.')

    else:
        print('No model to download.')
