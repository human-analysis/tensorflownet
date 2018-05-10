# imagefolder.py

import numpy as np
import os
from PIL import Image
import tensorflow as tf

class ImageFolder:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = []
        self.labels = []
        label = 0

        for _, folders, _ in os.walk(self.root):
            for folder in folders:
                for _, _, files in os.walk("{}/{}".format(self.root, folder)):
                    for file in files:
                        self.filenames.append("{}/{}/{}".format(self.root, folder, file))
                        self.labels.append(label)
                    break
                label += 1
            break

        self.length = len(self.filenames)
        print("Total = {}".format(self.length))

        dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.labels))
        dataset = dataset.map(lambda filename, label: tuple(tf.py_func(read_image_file, [filename, label], [tf.uint8, self.labels.dtype])))
        return dataset, self.length

def read_image_file(filename, label):
    with Image.open(filename) as image:
        im_numpy = np.array(image)
    return im_numpy, label
