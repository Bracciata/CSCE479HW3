from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from model import *
DATA_DIR = './tensorflow-datasets/'


ds=tfds.load(name="shapes3d", split="train", data_dir=DATA_DIR)

model = Model()
batch_size = 32
model.train(ds)
