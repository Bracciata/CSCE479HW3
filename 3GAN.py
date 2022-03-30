from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from tensorflow import keras

DATA_DIR = './tensorflow-datasets/'

ds = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test,y_test) = ds.load_data()
#train_ds = ds.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#train_ds = ds['train'].shuffle(1024).batch(32)


#for batch in train_ds:
    #print("data shape:", batch['image'].shape)
    #print("label:", batch['label'])
    #break

# visualize some of the data
#idx = np.random.randint(batch['image'].shape[0])
#print("An image looks like this:")
#imgplot = plt.imshow(batch['image'][idx])
#print(y_train[0])


codings_size = 30

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu",
    input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras. layers.Dense (28*28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])


discriminator = keras.models. Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(150,activation="selu"),
    keras.layers.Dense(100,activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator,discriminator])

discriminator.compile(loss="binary_crossentropy",optimizer= "rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True). prefetch(1)

def train_gan(gan,dataset,batch_size,codings_size,n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size,codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images,X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.1]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size,codings_size])
            y2 = tf.constant ([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            
train_gan(gan,dataset,batch_size,codings_size)




