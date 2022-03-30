from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
from tensorflow.keras import layers
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from keras_self_attention import SeqSelfAttention
from tensorflow import keras


class Model:
    # Updated from https://www.tensorflow.org/tutorials/generative/dcgan#the_generator
    def create_generator(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 256)))
        assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        # This adds color as described by https://keras.io/examples/generative/dcgan_overriding_train_step/#create-the-generator
        model.add(layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"))
        assert model.output_shape == (None, 64, 64, 3)
        return model
    def create_discriminator(self):
        model = tf.keras.Sequential()
        model.add( layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add( layers.LeakyReLU(alpha=0.2))
        model.add( layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(  layers.LeakyReLU(alpha=0.2))
        model.add(  layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(    layers.LeakyReLU(alpha=0.2))
        model.add(    layers.Flatten())
        model.add(   layers.Dropout(0.2))
        model.add(   layers.Dense(1, activation="sigmoid"))
        return model
    def discriminator_loss(self,real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    def generator_loss(self,fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        return cross_entropy(tf.ones_like(fake_output), fake_output)
    def __init__(self):
        self.generator = self.create_generator()
        self.discriminator =self.create_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    def train(self, dataset,codings_size=30, batch_size=32,n_epochs=50):
        dataset = dataset.batch(32)
        seed = tf.random.normal([1, 100])
        # Early stopping from https://www.tensorflow.org/guide/migrate/early_stopping
        patience = 5

        for epoch in range(n_epochs):
            print(epoch)
            wait = 0
            best = 0
            for batch in dataset:
                noise = tf.random.normal([batch_size, 100])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.generator(noise, training=True)

                    real_output = self.discriminator(tf.cast(batch['image'],tf.float32), training=True)
                    fake_output = self.discriminator(generated_images, training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                wait += 1
                if gen_loss > best:
                    best = gen_loss
                    wait = 0
                if wait >= patience:
                    break
            self.generate_and_save_images(self.generator,
                        epoch,
                        seed)
    def generate_and_save_images(self,model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(1, 1))

        for i in range(predictions.shape[0]):
            plt.subplot(1, 1, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()