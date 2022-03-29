from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from model import *
DATA_DIR = './tensorflow-datasets/'

# loading the dataset
MAX_SEQ_LEN = 128
MAX_TOKENS = 5000

ds = tfds.load(
    name="imdb_reviews", data_dir=DATA_DIR)
# Create TextVectorization layer
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode='int',
    output_sequence_length=MAX_SEQ_LEN)

# Use `adapt` to create a vocabulary mapping words to integers
train_text = ds['train'].map(lambda x: x['text'])
test_text = ds['test'].map(lambda x: x['text'])
vectorize_layer.adapt(train_text)
vectorize_layer.adapt(test_text)

ExplainBaseData(ds, vectorize_layer)
# Create the embedding layer
VOCAB_SIZE = len(vectorize_layer.get_vocabulary())
EMBEDDING_SIZE = int(np.sqrt(VOCAB_SIZE))
print("Vocab size is {} and is embedded into {} dimensions".format(
    VOCAB_SIZE, EMBEDDING_SIZE))

embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
# Create, train, and display the results from Model One
#modelOne = ModelOne(embedding_layer, vectorize_layer)
# modelOne.train(ds)
# modelOne.showResults()
# Create, train, and display the results from Model Two
# Reload the data to match the loading format for part two
# This part is heavily inspired by https://medium.com/@nutanbhogendrasharma/sentiment-classification-with-bidirectional-lstm-on-imdb-dataset-1ab21e6eeee9
# It aolso takes inspiration from many other tutorials
max_features = 9999
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_features
)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=140)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=140)
modelTwo = ModelTwo()
modelTwo.train(x_train, y_train, (x_test, y_test))
modelTwo.showResults()
