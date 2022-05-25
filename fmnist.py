import numpy as np
from urllib import request
import gzip
import pickle
from os import path
from torchvision.datasets import FashionMNIST

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def download_file(url, filename):
    opener = request.URLopener()
    opener.addheader('User-Agent', 'Mozilla/5.0')
    opener.retrieve(url, filename)


def download_fmnist():
  base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
  for name in filename:
    print("Downloading " + name[1] + "...")
    download_file(base_url + name[1], name[1])
    print("Download complete.")


def save_fmnist():
    fmnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            tmp = np.frombuffer(f.read(), np.uint8, offset=16)
            fmnist[name[0]] = tmp.reshape(-1, 1, 28, 28).astype(np.float32) / 255
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            fmnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("fmnist.pkl", 'wb') as f:
        pickle.dump(fmnist, f)
    print("Save complete.")


def init():
    # Check if already downloaded:
    if path.exists("fmnist.pkl"):
        print('Files already downloaded!')
    else:  # Download Dataset
        download_fmnist()
        save_fmnist()

    FashionMNIST(path.join('data', 'fmnist'), download=True)


def load():
    with open("fmnist.pkl", 'rb') as f:
        fmnist = pickle.load(f)
    return fmnist["training_images"], mnist["training_labels"], \
           fmnist["test_images"], mnist["test_labels"]


if __name__ == '__main__':
    init()
