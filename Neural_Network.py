import gzip
import numpy as np
import requests
import matplotlib.pyplot as plt


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
    f = gzip.open('train-images-idx3-ubyte.gz','r')
    f.read(16)
    data = np.frombuffer(f.read(image_dim * image_dim * image_cnt), dtype=np.uint8).astype(np.float32)
    data = data.reshape(image_cnt, 28, 28, 1)


def display_image(img_data,index):
    image_ndarray = np.asarray(img_data[index]).squeeze()
    plt.imshow(image_ndarray)
    plt.show()

