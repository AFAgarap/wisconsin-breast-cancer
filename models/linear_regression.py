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

"""Implementation of Linear Regression"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import time
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
                    output = tf.matmul(x_input, weight) + bias
                    output = tf.identity(output, name='output')
                    tf.summary.histogram('output', output)

            with tf.name_scope('mean_squared_loss'):
                loss = tf.reduce_mean(tf.square(output - y_onehot))
            tf.summary.scalar('loss', loss)

            train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = self.discretize(output)
                predicted_class = tf.identity(predicted_class, name='prediction')
                with tf.name_scope('correct_prediction'):
                    correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_onehot, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()

            self.x_input = x_input
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

        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(os.path.join(log_path, timestamp + '-training'),
                                             graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(os.path.join(log_path, timestamp + '-testing'),
                                            graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            try:
                for step in range(epochs * train_size // self.batch_size):
                    offset = (step * self.batch_size) % train_size
                    batch_train_data = train_data[0][offset:(offset + self.batch_size)]
                    batch_train_labels = train_data[1][offset:(offset + self.batch_size)]

                    feed_dict = {self.x_input: batch_train_data, self.y_input: batch_train_labels}

                    summary, _, step_loss, predicted, actual = sess.run([self.merged, self.train_op, self.loss,
                                                                         self.predicted_class, self.y_onehot],
                                                                        feed_dict=feed_dict)

                    if step % 100 == 0:
                        train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
                        print('step [{}] train -- loss : {}, accuracy : {}'.format(step, step_loss, train_accuracy))
                        train_writer.add_summary(summary=summary, global_step=step)

                    self.save_labels(predictions=predicted, actual=actual, result_path=result_path, step=step,
                                     phase='training')

            except KeyboardInterrupt:
                print('Training interrupted at step {}'.format(step))
                os._exit(1)
            finally:
                print('EOF -- training done at step {}'.format(step))

                for step in range(epochs * validation_size // self.batch_size):
                    offset = (step * self.batch_size) % validation_size
                    test_example_batch = validation_data[0][offset:(offset + self.batch_size)]
                    test_label_batch = validation_data[1][offset:(offset + self.batch_size)]

                    feed_dict = {self.x_input: test_example_batch, self.y_input: test_label_batch}

                    test_summary, predicted, actual = sess.run([self.merged, self.predicted_class, self.y_onehot],
                                                               feed_dict=feed_dict)

                    if step % 100 == 0 and step > 0:
                        test_accuracy, test_loss = sess.run([self.accuracy, self.loss], feed_dict=feed_dict)

                        print('step [{}] testing --- loss : {}, accuracy : {}'.format(step, test_loss, test_accuracy))

                        test_writer.add_summary(summary=test_summary, global_step=step)

                    self.save_labels(predictions=predicted, actual=actual, result_path=result_path, step=step,
                                     phase='testing')

                print('EOF -- testing done at step {}'.format(step))

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

        if not os.path.exists(path=result_path):
            os.mkdir(result_path)

        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)

        # save the labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-linear_regression-{}.npy'.format(phase, step)), arr=labels)

    @staticmethod
    def discretize(output):
        """Discretizes the predicted classes to [0, 1]

        Parameter
        ---------
        output : numpy.ndarray
          The NumPy array containing the predicted classes

        Returns
        -------
        output : numpy.ndarray
          The discretized predicted classes.
        """

        output = output >= 0.5
        output = tf.cast(output, 'float')
        return output
