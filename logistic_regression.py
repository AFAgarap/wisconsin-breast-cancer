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

import tensorflow as tf


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

    def train(self):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for step in range(1000):
                # train by batch of 100
                batch_xs, batch_ys = self.data_input.train.next_batch(100)
                
                # input dictionary
                feed_dict = {self.x: batch_xs, self.y: batch_ys}
                
                # run the train operation
                _ = sess.run([self.train_op], feed_dict=feed_dict)
                
                # every 100th step and at step 0,
                # display the loss and accuracy of the model
                if step % 100 == 0:
                    loss, accuracy = sess.run([self.cross_entropy, self.accuracy_op], feed_dict=feed_dict)

                    print('step [{}] -- loss: {}, accuracy: {}'.format(step, loss, accuracy))

            feed_dict = {self.x: self.data_input.test.images, self.y: self.data_input.test.labels}
            
            # get the accuracy of the train model
            # using unseen data
            test_accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            print('Test Accuracy: {}'.format(test_accuracy))

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
