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

"""Implementation of Support Vector Machine using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.5'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import sys
import tensorflow as tf
import time


class SVM:
    """Implementation of L2-Support Vector Machine using TensorFlow"""

    def __init__(self, alpha, batch_size, svm_c, num_classes, num_features):
        """Initialize the SVM class

        Parameter
        ---------
        alpha : float
          The learning rate for the SVM model.
        batch_size : int
          Number of batches to use for training and testing.
        svm_c : float
          The SVM penalty parameter.
        num_classes : int
          Number of classes in a dataset.
        num_features : int
          Number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.svm_c = svm_c
        self.num_classes = num_classes
        self.num_features = num_features

        def __graph__():
            """Building the inference graph"""

            with tf.name_scope('input'):

                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_features], name='x_input')

                # [BATCH_SIZE]
                y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name='y_input')

                # [BATCH_SIZE, NUM_CLASSES]
                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1, off_value=-1,
                                      name='y_onehot')

            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.get_variable(name='weights',
                                             initializer=tf.random_normal([self.num_features, self.num_classes],
                                                                          stddev=0.01))
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.get_variable(name='biases', initializer=tf.constant([0.1], shape=[self.num_classes]))
                    self.variable_summaries(bias)
                with tf.name_scope('Wx_plus_b'):
                    output = tf.matmul(x_input, weight) + bias
                    tf.summary.histogram('pre-activations', output)

            with tf.name_scope('svm'):
                regularization = 0.5 * tf.reduce_sum(tf.square(weight))
                hinge_loss = tf.reduce_sum(
                    tf.square(tf.maximum(tf.zeros([self.batch_size, self.num_classes]),
                                         1 - tf.cast(y_onehot, tf.float32) * output)))
                with tf.name_scope('loss'):
                    loss = regularization + self.svm_c * hinge_loss
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

            with tf.name_scope('accuracy'):
                predicted_class = tf.sign(output)
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
            self.learning_rate = learning_rate
            self.loss = loss
            self.optimizer = optimizer
            self.output = output
            self.predicted_class = predicted_class
            self.accuracy = accuracy
            self.merged = merged

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, epochs, log_path, train_data, train_size, validation_data, validation_size, result_path):
        """Trains the SVM model

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

        # initialize the variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # get the current time and date
        timestamp = str(time.asctime())

        # event files to contain the TensorBoard log
        train_writer = tf.summary.FileWriter(log_path + timestamp + '-training', graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            try:
                for step in range(epochs * train_size // self.batch_size):
                    offset = (step * self.batch_size) % train_size
                    batch_train_data = train_data[0][offset:(offset + self.batch_size)]
                    batch_train_labels = train_data[1][offset:(offset + self.batch_size)]

                    feed_dict = {self.x_input: batch_train_data, self.y_input: batch_train_labels,
                                 self.learning_rate: self.alpha}

                    summary, _, step_loss = sess.run([self.merged, self.optimizer, self.loss], feed_dict=feed_dict)

                    if step % 100 == 0:
                        train_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
                        print('step[{}] train -- loss : {}, accuracy : {}'.format(step, step_loss, train_accuracy))
                        train_writer.add_summary(summary=summary, global_step=step)
            except KeyboardInterrupt:
                print('Training interrupted at step {}'.format(step))
                os._exit(1)
            finally:
                print('EOF -- training done at step {}'.format(step))

                for step in range(epochs * validation_size // self.batch_size):

                    feed_dict = {self.x_input: validation_data[0][:self.batch_size],
                                 self.y_input: validation_data[1][:self.batch_size]}

                    validation_summary, predictions, actual, validation_loss, validation_accuracy = \
                        sess.run([self.merged, self.predicted_class, self.y_onehot, self.loss, self.accuracy],
                                 feed_dict=feed_dict)

                    if step % 100 == 0 and step > 0:
                        print('step [{}] validation -- loss : {}, accuracy : {}'.format(step, validation_loss,
                                                                                        validation_accuracy))

                    self.save_labels(predictions=predictions, actual=actual, result_path=result_path, step=step,
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

        # Concatenate the predicted and actual labels
        labels = np.concatenate((predictions, actual), axis=1)

        # save the labels array to NPY file
        np.save(file=os.path.join(result_path, '{}-svm-{}.npy'.format(phase, step)), arr=labels)
