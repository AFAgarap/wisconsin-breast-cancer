# Copyright 2017 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""Utility functions for data handling"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.2.0'
__author__ = 'Abien Fred Agarap'

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf


def list_files(path):
    """Returns a list of files

    Parameter
    ---------
    path : str
      A string consisting of a path containing files.

    Returns
    -------
    file_list : list
      A list of the files present in the given directory

    Examples
    --------
    >>> PATH = '/home/data'
    >>> list_files(PATH)
    >>> ['/home/data/file1', '/home/data/file2', '/home/data/file3']
    """

    file_list = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        file_list.extend(os.path.join(dir_path, filename) for filename in file_names)
    return file_list


def plot_confusion_matrix(phase, path, class_names):
    """Plots the confusion matrix using matplotlib.

    Parameter
    ---------
    phase : str
      String value indicating for what phase is the confusion matrix, i.e. training/validation/testing
    path : str
      Directory where the predicted and actual label NPY files reside
    class_names : str
      List consisting of the class names for the labels

    Returns
    -------
    conf : array, shape = [num_classes, num_classes]
      Confusion matrix
    accuracy : float
      Predictive accuracy
    """

    # list all the results files
    files = list_files(path=path)

    labels = np.array([])

    for file in files:
        labels_batch = np.load(file)
        labels = np.append(labels, labels_batch)

        if (files.index(file) / files.__len__()) % 0.2 == 0:
            print('Done appending {}% of {}'.format((files.index(file) / files.__len__()) * 100, files.__len__()))

    labels = np.reshape(labels, newshape=(labels.shape[0] // 4, 4))

    print('Done appending NPY files.')

    # get the predicted labels
    predictions = labels[:, :2]

    # get the actual labels
    actual = labels[:, 2:]

    # create a TensorFlow session
    with tf.Session() as sess:

        # decode the one-hot encoded labels to single integer
        predictions = sess.run(tf.argmax(predictions, 1))
        actual = sess.run(tf.argmax(actual, 1))

    # get the confusion matrix based on the actual and predicted labels
    conf = confusion_matrix(y_true=actual, y_pred=predictions)

    # create a confusion matrix plot
    plt.imshow(conf, cmap=plt.cm.Purples, interpolation='nearest')

    # set the plot title
    plt.title('Confusion Matrix for {} Phase'.format(phase))

    # legend of intensity for the plot
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # show the plot
    plt.show()

    # get the accuracy of the phase
    accuracy = (conf[0][0] + conf[1][1]) / labels.shape[0]

    # return the confusion matrix and the accuracy
    return conf, accuracy


def get_statistical_measures(conf_matrix):
    """Returns an array of statistical measures

    Parameter
    ---------
    conf_matrix : array
      The confusion matrix

    Returns
    -------
    statistical_measures : numpy.ndarray
      The NumPy array containing the statistical measures
    """
    true_positive_rate = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    true_negative_rate = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    false_positive_rate = 1 - true_negative_rate
    false_negative_rate = 1 - true_positive_rate
    statistical_measures = np.array([true_negative_rate, true_positive_rate, false_negative_rate, false_positive_rate])
    return statistical_measures
