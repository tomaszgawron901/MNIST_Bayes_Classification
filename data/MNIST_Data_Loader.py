import requests
import struct
import os.path
import gzip
import numpy as np

_training_imgs_name = 'train-images-idx3-ubyte'
_training_labels_name = 'train-labels-idx1-ubyte'
_test_imgs_name = 't10k-images-idx3-ubyte'
_test_labels_name = 't10k-labels-idx1-ubyte'

_MNISTDataURL = "http://yann.lecun.com/exdb/mnist/"


def get_training_data(directory: str):
    """
    Returns tuple containing 60k training images and labels.
    """
    return _get_imgs(directory, _training_imgs_name), _get_labels(directory, _training_labels_name)


def get_test_data(directory: str):
    """
    Returns tuple containing 10k test images and labels.
    """
    return _get_imgs(directory, _test_imgs_name), _get_labels(directory, _test_labels_name)


def _download_data(url: str, directory: str, decompress: bool = True):
    """
    Downloads data from url address, decompresses it (if needed) and saves it to given directory.
    """
    data = requests.get(url, allow_redirects=True).content
    if decompress:
        data = gzip.decompress(data)
    file = open(directory, "wb")
    file.write(data)
    file.close()


def _get_file(directory: str, name: str):
    """
    Opens file of given name.
    """
    file_path = os.path.join(directory, name)
    if not os.path.exists(file_path):
        url = _MNISTDataURL + name + '.gz'
        _download_data(url, file_path)
    return open(file_path, 'rb')


def _get_labels(directory: str, name: str):
    file = _get_file(directory, name)
    magic_number, size = struct.unpack(">II", file.read(8))
    if magic_number != 2049:
        raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic_number))
    labels: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)
    file.close()
    return labels


def _get_imgs(directory: str, name: str):
    file = _get_file(directory, name)
    magic_number, size, rows, columns = struct.unpack(">IIII", file.read(16))
    if magic_number != 2051:
        raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic_number))
    images: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)
    file.close()
    images.resize((size, rows*columns))
    return images.astype(np.float) / 255
