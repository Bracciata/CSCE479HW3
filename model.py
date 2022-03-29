from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util import *
from keras_self_attention import SeqSelfAttention


class ModelOne:
    def __init__(self, embedding_layer, vectorize_layer):
        self.embedding_layer = embedding_layer
        self.vectorize_layer = vectorize_layer

        L2_COEFF = 0.1  # Controls how strongly to use regularization
        # Defining, creating and calling the network repeatedly will trigger a WARNING about re-tracing the function
        # So we'll check to see if the variable exists already
        if 'l2_dense_net' not in locals():
            l2_dense_net = L2DenseNetwork()

        # We'll make a conv layer to produce the query and value tensors
        self.query_layer = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=4,
            padding='same')
        self.value_layer = tf.keras.layers.Conv1D(
            filters=100,
            kernel_size=4,
            padding='same')
        # Then they will be input to the Attention layer
        self.attention = tf.keras.layers.Attention()
        self.concat = tf.keras.layers.Concatenate()

        cells = [tf.keras.layers.LSTMCell(256), tf.keras.layers.LSTMCell(64)]
        rnn = tf.keras.layers.RNN(cells)
        output_layer = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = tf.keras.Sequential(
            [self.vectorize_layer, self.embedding_layer, rnn, output_layer])
        self.loss_values = []
        self.accuracy_values = []

    def train(self, ds):
        for epoch in range(1):
            count = 0
            for batch in ds['test'].batch(32):
                with tf.GradientTape() as tape:
                    text = batch['text']
                    labels = batch['label']

                    embeddings = self.embedding_layer(
                        self.vectorize_layer(text))
                    query = self.query_layer(embeddings)
                    value = self.value_layer(embeddings)
                    query_value_attention = self.attention([query, value])
                    print("Shape after attention is (batch, seq, filters):",
                          query_value_attention.shape)
                    attended_values = self.concat(
                        [query, query_value_attention])
                    print("Shape after concatenating is (batch, seq, filters):",
                          attended_values.shape)
                    logits = self.model(batch['text'])
                    loss = tf.keras.losses.binary_crossentropy(
                        tf.expand_dims(batch['label'], -1), logits, from_logits=True)
                    print()
                    print("Loss values in Training set:")
                    print(loss)

                self.loss_values.append(loss)
                # gradient update
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, labels), tf.float32))
                self.accuracy_values.append(accuracy)
                count += 1
                if count > 50:
                    break

    def showResults(self):
        print(self.model.summary())
        # plot per-datum loss
        # loss_values = np.concatenate(loss_values)
        plt.hist(self.loss_values, density=True, bins=30)

        plt.plot(self.loss_values)

        plt.show()


class ModelTwo:
    def __init__(self):
        inputs = tf.keras.Input(shape=(None,), dtype="int32")
        embedding_layer = tf.keras.layers.Embedding(
            9999, 140)
        model = tf.keras.Sequential()
        model.add(inputs)
        model.add(embedding_layer)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            128, return_sequences=True)))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            32)))
        model.add(tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=0.001, l2=0.001), activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        self.model = model
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        # Early Stopping Callback from https://keras.io/api/callbacks/early_stopping/
        self.callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3)

    def train(self, xTrain, yTrain, validation):
        self.history = self.model.fit(xTrain, yTrain,
                                      batch_size=64,
                                      epochs=5,
                                      validation_data=validation,
                                      verbose=1, callbacks=[self.callback])

    def showResults(self):
        print(self.model.summary())
        plt.title('Loss over each Epoch')
        plt.plot(self.history.history['loss'])
        plt.savefig('lossRunOne.png')

        plt.title('Accuracy over each Epoch')
        plt.plot(self.history.history['accuracy'])
        plt.savefig('accuracyRunOne.png')
