import os
import urllib.request
import tarfile
from tqdm import tqdm

class DLProgress(tqdm):
    def hook(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download_and_extract_model(model_name):
    """
    Download and extract model if it doesn't exist already
    :param model_name: The model name to download from 'http://download.tensorflow.org/models/object_detection/'
    :return:
    """
    if model_name:
        # Set complete URL to target for downloading model
        base = 'http://download.tensorflow.org/models/object_detection/'
        file = model_name + '.tar.gz'
        url = base + file

        # Download model
        if not os.path.exists(model_name):
            print('Downloading model {0}...'.format(model_name))
            with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
                url_opener = urllib.request.URLopener()
                url_opener.retrieve(url, file, pbar.hook)

            # Extract tar file
            print('Extracting model archive {0}...'.format(file))
            tar_file = tarfile.open(file, 'r:gz')
            tar_file.extractall()

            # Remove tar file to save space
            os.remove(file)

    else:
        print('No model to download.')
