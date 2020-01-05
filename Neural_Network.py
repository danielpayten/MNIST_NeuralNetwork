import gzip
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def download_files():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        raw_file = requests.get('http://yann.lecun.com/exdb/mnist/' + file)
        with open(file, 'wb') as f:
            f.write(raw_file.content)


def preprocess_images():
    image_dim = 28
    image_cnt = 60000
    f = gzip.open('train-images-idx3-ubyte.gz', 'r')
    # Skip the first 16 lines (preamble)
    f.read(16)
    data = np.frombuffer(f.read(image_dim * image_dim * image_cnt), dtype=np.uint8).astype(np.float32)
    data = data.reshape(image_cnt, image_dim*image_dim)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = scaled_data.reshape(image_cnt, image_dim, image_dim)
    return scaled_data

def preprocess_labs():
    image_cnt = 60000
    f = gzip.open('train-labels-idx1-ubyte.gz', 'r')
    # Skip the first 8 lines (preamble)
    f.read(8)
    labs = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int)
    labs = labs.reshape(image_cnt, 1)

    # rather than having labs formatted as a vector of integers, need logical row vector for each
    # all zeros with a 1 in the relevant digit column.
    y_mat = np.zeros( (len(labs), 10))
    integer_array = np.arange(10)

    for i, row in enumerate(labs):
        logic_matrix = np.asarray(integer_array == row).astype(np.int64)
        y_mat[i] = logic_matrix
    return y_mat


def display_image(img_data, index):
    image_ndarray = np.asarray(img_data[index]).squeeze()
    plt.imshow(image_ndarray)
    plt.show()









