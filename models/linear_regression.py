# MIT License
# 
# Copyright (c) 2017 Abien Fred Agarap
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Implementation of Linear Regression"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import tensorflow as tf
import sys


class LinearRegression:
    """Implementation of the Linear Regression algorithm using TensorFlow"""

    def __init__(self, alpha, batch_size, num_classes, sequence_length):
        """Initialize the Linear Regression class

        Parameter
        ---------
        alpha : float
          The learning rate for the Linear Regression model.
        batch_size : int
          The number of batches to use for training/validation/testing.
        num_classes : int
          The number of classes in a dataset.
        sequence_length : int
          The number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        def __graph__():

            with tf.name_scope('input'):
                # [BATCH_SIZE, SEQUENCE_LENGTH]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.sequence_length], name='x_input')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name='y_input')

                # [BATCH_SIZE, NUM_CLASSES]
                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1.0, off_value=0.0,
                                      name='y_onehot')

            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.Variable(tf.zeros([self.sequence_length, self.num_classes]), name='weight')
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.Variable(tf.zeros([self.num_classes]), name='bias')
                    self.variable_summaries(bias)
                with tf.name_scope('decision_function'):
                    output = tf.matmul(weight, x_input) + bias
                    output = tf.identity(output, name='output')
                    tf.summary.histogram('output', output)

            loss = tf.reduce_sum(tf.square(output - y_input))

            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = 1 if output >= 0.5 else 0
                predicted_class = tf.identity(predicted_class, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_onehot, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()

            self.weight = weight
            self.x_input = x_input
            self.bias = bias
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.output = output
            self.loss = loss
            self.train_op = train_op
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, epochs, log_path, train_data, train_size, validation_data, validation_size, result_path):
        """Trains the Linear Regression model

        Parameter
        ---------
        epochs : int
          The number of passes through the entire dataset.
        log_path : str
          The directory where to save the TensorBoard logs.
        train_data : numpy.ndarray
          The numpy.ndarray to be used as the training dataset.
        train_size : int
          The number of data in `train_data`.
        validation_data : numpy.ndarray
          The numpy.ndarray to be used as the validation dataset.
        validation_size : int
          The number of data in `validation_data`.
        result_path : str
          The path where to save the NPY files consisting of the actual and predicted labels.
        """

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)

            for step in range(1000):
                # load the data
                x_train, y_train = self.data_input
                
                # create input dictionary to feed to the train operation
                feed_dict = {self.x_input: x_train, self.y: y_train}
                
                # run the train operation with the previously-defined input dict
                sess.run(self.train_op, feed_dict=feed_dict)
                
                # get the learnt parameters and the error (loss)
                curr_w, curr_b, curr_loss = sess.run([self.weight, self.bias, self.loss], feed_dict=feed_dict)
                
            # print the learn parameters of the regression and its loss
            print("W: {}, b: {}, loss: {}".format(curr_w, curr_b, curr_loss))

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def save_labels(predictions, actual, result_path, step, phase):
        """Saves the actual and predicted labels to a NPY file

        Parameter
        ---------
        predictions : numpy.ndarray
          The NumPy array containing the predicted labels.
        actual : numpy.ndarray
          The NumPy array containing the actual labels.
        result_path : str
          The path where to save the concatenated actual and predicted labels.
        step : int
          The time step for the NumPy arrays.
        phase : str
          The phase for which the predictions is, i.e. training/validation/testing.
        """

        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)

        # save the labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-linear_regression-{}.npy'.format(phase, step)), arr=labels)
