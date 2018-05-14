import numpy as np
import tensorflow as tf
import utils
import math

class CNN_model:
    def __init__(self, data, n_labels, labels, n_filters):

        self._data = data
        self._n_labels = n_labels
        self._labels = labels

        conv_layer = data

        for n in n_filters:
            conv_layer = self.conv_pool(n, conv_layer)

        flat = tf.reshape(conv_layer, [-1, 24 * 24 * n_filters[-1]])
        self._dense = tf.layers.dense(inputs=flat,
                                      units=256,
                                      activation=tf.nn.relu,)

        self._predict = None
        self._loss = None
        self._optimizer = None

        self.predict
        self.loss
        self.optimizer

    def conv_pool(self, nfilters, x):

        conv = tf.layers.conv2d(inputs= x,
                                filters= nfilters,
                                kernel_size= utils.FSIZE,
                                padding= "same",
                                activation= tf.nn.relu)

        pool = tf.layers.max_pooling2d(inputs = conv,
                                       pool_size = utils.PSIZE,
                                       strides = utils.STRIDE)

        return pool

    @property
    def loss(self):
        if self._loss is None:
            self._loss = tf.sqrt(
                tf.losses.mean_squared_error(
                    labels=self._labels,
                    predictions=self.predict)
            )
        return self._loss

    @property
    def predict(self):
        if self._predict is None:
            self._predict = tf.layers.dense(inputs=self._dense, units=30)
        return self._predict

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        return self._optimizer
