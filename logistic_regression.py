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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import os
import tensorflow as tf
import time


class LogisticRegression:
    """Implementation of the Logistic Regression algorithm using TensorFlow"""
    def __init__(self, alpha, batch_size, cell_size, dropout_rate, num_classes, sequence_length, svm_c):
        """Initialize the Logistic Regression class

        Parameter
        ---------
        alpha : float
          The learning rate for the GRU+Softmax model.
        batch_size : int
          The number of batches to use for training/validation/testing.
        num_classes : int
          The number of classes in a dataset.
        sequence_length : int
          The number of features in a dataset.
        """
        self.alpha = alpha
        self.batch_Size = batch_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        def __graph__():
            # input placeholder for features (x) and labels (y)
            with tf.name_scope('input'):
                x_input = tf.placeholder(tf.float32, [None, self.sequence_length], name='x_input')

                y_input = tf.placeholder(tf.uint8, [None], name='y_input')

                y_onehot = tf.one_hot(indices=y_input, depth=self.num_classes, on_value=1.0, off_value=-1.0,
                                        name='y_onehot')
            
            with tf.name_scope('training_ops'):
                with tf.name_scope('weights'):
                    weight = tf.Variable(tf.zeros([self.sequence_length, self.num_classes]), name='weights')
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.Variable(tf.zeros([self.num_classes]), name='biases')
                    self.variable_summaries(bias)
                with tf.name_scope('decision_function'):
                    output = tf.matmul(x_input, weight) + bias
                    output = tf.identity(output, name='logits')
                    tf.summary.histogram('pre-activations', output)
                        
            with tf.name_scope('cross_entropy_loss'):
                # get the loss of the training
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=output))
            tf.summary.scalar('loss', cross_entropy)

            # train using SGD algorithm
            train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
            
            with tf.name_scope('accuracy'):
                # get the predicted probability distribution
                predictions = tf.nn.softmax(y, name='predictions')
                with tf.name_scope('correct_predition'):
                    # check if the actual labels and predicted labels match
                    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y_onehot, 1))
                with tf.name_scope('accuracy'):
                    # get the % of correct predictions
                    accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            self.x_input = x_input
            self.y_input = y_input
            self.y_onehot = y_onehot
            self.output = output
            self.predictions = predictions
            self.cross_entropy = cross_entropy
            self.train_op = train_op
            self.accuracy_op = accuracy_op

        sys.stdout.write('\n<log> Building graph...')
        __graph__()
        sys.stdout.write('</log>\n')

    def train(self, checkpoint_path, log_path, model_name, epochs, train_data, train_size, validation_data,
              validation_size, result_path):
        """Trains the model

        Parameter
        ---------
        checkpoint_path : str
          The path where to save the trained model.
        log_path : str
          The path where to save the TensorBoard summaries.
        model_name : str
          The filename for the trained model.
        epochs : int
          The number of passes through the whole dataset.
        train_data : numpy.ndarray
          The NumPy array training dataset.
        train_size : int
          The size of `train_data`.
        validation_data : numpy.ndarray
          The NumPy array testing dataset.
        validation_size : int
          The size of `validation_data`.
        result_path : str
          The path where to save the actual and predicted classes array.
        """

        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        saver = tf.train.Saver(max_to_keep=10)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        timestamp = str(time.asctime())

        train_writer = tf.summary.FileWriter(logdir=log_path + timestamp + '-training', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logdir=log_path + timestamp + '-testing', graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init_op)

            checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path)
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

            try:
                for step in range(epochs * train_size // self.batch_size):
                    offset = (step * self.batch_size) % train_size
                    train_example_batch = train_data[0][offset:(offset + self.batch_size)]
                    train_label_batch = train_data[1][offset:(offset + self.batch_size)]

                    feed_dict = {self.x_input: train_example_batch, self.y_input: train_label_batch}

                    _, loss, accuracy, predicted, actual = sess.run([self.train_op, self.loss, self.accuracy_op,
                                                                     self.predictions, self.y_onehot],
                                                                     feed_dict=feed_dict)
                    if step % 100 == 0 and step > 0:
                        
            except KeyboardInterrupt:
                print('Training interrupted at step {}'.format(step))
                os._exit(1)
            finally:
                print('EOF -- Training done at step {}'.format(step))


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
