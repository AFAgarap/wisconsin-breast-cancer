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

"""An implementation of K-Nearest Neighbor based on [1]

[1] Aymeric Damien. 2017, August 29. (2017, August 29).
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/nearest_neighbor.py
Accessed: November 17, 2017.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import tensorflow as tf


class NearestNeighbor:

    def __init__(self, train_features, train_labels, sequence_length):
        self.train_features = train_features
        self.train_labels = train_labels

        with tf.name_scope('input'):
            xtr = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length])
            xte = tf.placeholder(dtype=tf.float32, shape=[sequence_length])

        # L1-Norm
        # distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

        # L2-Norm
        distance = tf.sqrt(tf.reduce_sum(tf.square(xtr - xte), reduction_indices=1))

        prediction = tf.arg_min(distance, 0)

        accuracy = 0.

        self.xtr = xtr
        self.xte = xte
        self.distance = distance
        self.prediction = prediction
        self.accuracy = accuracy

    def predict(self, test_features, test_labels, result_path):

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

                self.save_labels(predictions=np.argmax(y[nn_index]), actual=np.argmax(y_[index]),
                                 result_path=result_path, step=index, phase='testing')

                if np.argmax(y[nn_index]) == np.argmax(y_[index]):
                    self.accuracy += 1. / len(test_features)

        print('Accuracy : {}'.format(self.accuracy))

    @staticmethod
    def save_labels(predictions, actual, result_path, step, phase):
        """Saves the actual and predicted labels to a NPY file

        Parameter
        ---------
        predictions : int
          The predicted label.
        actual : int
          The actual label.
        result_path : str
          The path where to save the concatenated actual and predicted labels.
        step : int
          The time step for the predicted and actual labels
        phase : str
          The phase for which the prediction is, i.e. training/validation/testing.
        """

        if not os.path.exists(path=result_path):
            os.mkdir(result_path)

        # Concatenate the predicted and actual labels
        labels = np.array([predictions, actual])

        # save the labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-nearest_neighbor-{}.npy'.format(phase, step)), arr=labels)
