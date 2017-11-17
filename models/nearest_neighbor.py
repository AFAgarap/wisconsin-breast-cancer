# Copyright 2017 Abien Fred Agarap
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An implementation of K-Nearest Neighbor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf


class NearestNeighbor:

    def __init__(self, train_features, train_labels, sequence_length):
        self.train_features = train_features
        self.train_labels = train_labels

        with tf.name_scope('input'):
            xtr = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length])
            xte = tf.placeholder(dtype=tf.float32, shape=[sequence_length])

        distance = tf.sqrt(tf.reduce_sum(tf.square(xtr - xte), reduction_indices=1))

        prediction = tf.arg_min(distance, 0)

        accuracy = 0.

        self.xtr = xtr
        self.xte = xte
        self.distance = distance
        self.prediction = prediction
        self.accuracy = accuracy

    def predict(self, test_features, test_labels):

        train_labels = tf.one_hot(self.train_labels, depth=2, on_value=1, off_value=0)
        test_labels = tf.one_hot(test_labels, depth=2, on_value=1, off_value=0)

        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            y_, y = sess.run([test_labels, train_labels])

            # loop over test data
            for index in range(len(test_features)):

                feed_dict = {self.xtr: self.train_features, self.xte: test_features[index, :]}

                nn_index = sess.run(self.prediction, feed_dict=feed_dict)

                print('Test [{}] Actual Class: {}, Predicted Class : {}'.format(index, np.argmax(y_[index]),
                                                                                np.argmax(y[nn_index])))

                if np.argmax(y[nn_index]) == np.argmax(y_[index]):
                    self.accuracy += 1. / len(test_features)

        print('Accuracy : {}'.format(self.accuracy))
