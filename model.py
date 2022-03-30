from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from keras_self_attention import SeqSelfAttention
from tensorflow import keras


class Model:
    def __init__(self):
        codings_size = 64

        self.generator = keras.models.Sequential([
            keras.layers.Dense(100, activation="selu",
            input_shape=(codings_size,codings_size,3)),
            keras.layers.Dense(150, activation="selu"),
            keras.layers.Dense (3, activation="sigmoid"),
            keras.layers.Reshape([64,64,3])
        ])


        self.discriminator = keras.models.Sequential([
            keras.layers.Flatten(input_shape=[64,64,3]),
            keras.layers.Dense(150,activation="selu",input_shape=(codings_size,codings_size,3)),
            keras.layers.Dense(100,activation="selu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        self.gan = keras.models.Sequential([self.generator,self.discriminator])

        self.discriminator.compile(loss="binary_crossentropy",optimizer= "rmsprop")
        self.discriminator.trainable = False
        self.gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    def train(self, dataset,codings_size=30, batch_size=32,n_epochs=50):
       
        for epoch in range(n_epochs):
            print(epoch)
            
            for X_batch in dataset:
                print(X_batch)
                print(X_batch['image'])
                # phase 1 - training the discriminator
                noise = tf.random.normal(shape=[64,64,3])
                generated_images = tf.squeeze(self.generator(tf.expand_dims(noise,axis=0)),axis=0)
                images = tf.cast(X_batch['image'],tf.float32)
                #X_fake_and_real = tf.concat([generated_images,x], axis=0)
                
                y1 = tf.constant([[0.]] * batch_size + [[1.1]] * batch_size)
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(tf.expand_dims(generated_images,axis=0))
                self.discriminator.train_on_batch(tf.expand_dims(images,axis=0))
                # phase 2 - training the generator
                noise = tf.random.normal(shape=[batch_size,codings_size])
                y2 = tf.constant ([[1.]] * batch_size)
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise, y2)

   