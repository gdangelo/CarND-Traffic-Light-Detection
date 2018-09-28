import os
import urllib.request
import tarfile
from tqdm import tqdm
import tensorflow as tf
from lxml import etree
from object_detection.utils import dataset_util

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

def get_data_from_xml(path):
    """
    Parse XML file and retrieve labels data needed to create TF record
    :param path: Path to XML file
    :return:
        - xmins: List of normalized left x coordinates in bounding box (1 per box)
        - xmaxs: List of normalized right x coordinates in bounding box # (1 per box)
        - ymins: List of normalized top y coordinates in bounding box (1 per box)
        - ymaxs: List of normalized bottom y coordinates in bounding box # (1 per box)
        - classes_text: # List of string class name of bounding box (1 per box)
        - classes: # List of integer class id of bounding box (1 per box)
        - width: Image width
        - height: Image height
    """
    # Create label map dict
    label_map = {'green':1, 'yellow':2, 'red':3, 'nolight':4}

    # Read XML file
    with tf.gfile.GFile(path, "rb") as file:
      xml_str = file.read()

    # Parse XML data
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    # Get labels data
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmins.append(float(obj['bndbox']['xmin']) / width)
        xmaxs.append(float(obj['bndbox']['xmax']) / width)
        ymins.append(float(obj['bndbox']['ymin']) / height)
        ymaxs.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map[obj['name']])

    return xmins, xmaxs, ymins, ymaxs, classes_text, classes, width, height

def create_tf_example(data):
    """
    Create TF example from an image file and its corresponding labels
    :param data: Image and labels files path
    :return: tf_example: TF example
    """
    # Set image properties
    image_format = b'jpeg'
    filename = data['image']['filename'].encode('utf8')

    # Read JPG image file
    with tf.gfile.GFile(data['image']['path'], "rb") as file:
      encoded_image_data = file.read() # Encoded image bytes

    # Get data from labels XML file
    xmins, xmaxs, ymins, ymaxs, classes_text, classes, width, height = get_data_from_xml(data['labels']['path'])

    # Create TF example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(images_dir, labels_dir, output_path):
    """
    Create TF record using images in images_dir/ and corresponding XML labels
    files under labels_dir/.
    :param images_dir: Images directory (JPG files)
    :param labels_dir: Labels directory (PascalVOC's XML files)
    :param output_path: TF record output path
    """

    writer = tf.python_io.TFRecordWriter(output_path)

    for entry in os.listdir(images_dir):
        # Check if entry in images_dir/ is a subdirectory
        data_dir = os.path.join(images_dir, entry)
        if os.path.isdir(data_dir):

            # Process each image file
            for filename in os.listdir(data_dir):

                # Check if corresponding labels file exist
                labels_filename = filename.replace('jpg', 'xml')
                if os.path.exists(os.path.join(labels_dir, labels_filename)):
                    # Build dict to store image and labels files path
                    data = {
                        'image': {
                            'path': os.path.join(data_dir, filename),
                            'filename': filename
                        },
                        'labels': {
                            'path': os.path.join(labels_dir, labels_filename)
                        }
                    }

                    # Create TF example based for each image
                    tf_example = create_tf_example(data)
                    writer.write(tf_example.SerializeToString())

    writer.close()

def convert_dataset_into_tfrecord():
    """
    Convert training and eval dataset into TF records.
    Labels are stored in XML files based on PascalVOC annotation.

    Dataset must have the following folders structure:
    - data/
        - train/
            - images/
                + green/*.jpg
                + yellow/*.jpg
                + red/*.jpg
                + nolight/*.jpg
            - labels/*.xml
        - eval/
            + ...
    """
    dataset_dir = 'data/'

    train_images_dir = os.path.join(dataset_dir, 'train/', 'images/')
    train_labels_dir = os.path.join(dataset_dir, 'train/', 'labels/')
    train_output_tf_record = os.path.join(dataset_dir, './train_traffic_light_udacity_sim.record')

    eval_images_dir = os.path.join(dataset_dir, 'eval/', 'images/')
    eval_labels_dir = os.path.join(dataset_dir, 'eval/', 'labels/')
    eval_output_tf_record = os.path.join(dataset_dir, './eval_traffic_light_udacity_sim.record')

    print('Creating TF record for training dataset...')
    create_tf_record(train_images_dir, train_labels_dir, train_output_tf_record)
    print('Creating TF record for evaluation dataset...')
    create_tf_record(eval_images_dir, eval_labels_dir, eval_output_tf_record)

if __name__ == '__main__':
    #download_and_extract_model('ssd_inception_v2_coco_2018_01_28')
    convert_dataset_into_tfrecord()
