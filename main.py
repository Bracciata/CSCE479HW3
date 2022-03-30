from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from model import *
import os.path 

if not os.path.isdir('celeba_gan'):
    download_celebs() 

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
ds = dataset.map(lambda x: x / 255.0)


model = Model()
batch_size = 32
model.train(ds)
