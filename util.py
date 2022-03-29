from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


class L2DenseNetwork(tf.Module):
    def __init__(self, name=None):
        # remember this call to initialize the superclass
        super(L2DenseNetwork, self).__init__(name=name)
        self.dense_layer1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense_layer2 = tf.keras.layers.Dense(10)

    def l2_loss(self):
        # Make sure the network has been called at least once to initialize the dense layer kernels
        return tf.nn.l2_loss(self.dense_layer1.kernel) + tf.nn.l2_loss(self.dense_layer2.kernel)

    @tf.function
    def __call__(self, x):
        embed = self.dense_layer1(x)
        output = self.dense_layer2(embed)
        return output


def ExplainBaseData(ds, vectorize_layer):
    # Let's print out a batch to see what it looks like in text and in integers
    print("This review is from training set:")
    for batch in ds['train'].batch(1):
        text = batch['text']
        label = batch['label']
        #print(list(zip(text.numpy(), vectorize_layer(text).numpy())))
        print()
        print("Length of vector is: ", vectorize_layer(text).shape)
        print("The label for this review is: ", label.numpy())
        break

    print()
