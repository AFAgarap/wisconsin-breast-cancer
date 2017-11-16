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

"""Main program using the SVM class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.5'
__author__ = 'Abien Fred Agarap'

import argparse
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import svm
import utils

BATCH_SIZE = 40
LEARNING_RATE = 1e-3
NUM_CLASSES = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description='SVM built using TensorFlow, for Wisconsin Breast Cancer Diagnostic Dataset')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-c', '--svm_c', required=True, type=int,
                       help='Penalty parameter C of the SVM')
    group.add_argument('-n', '--num_epochs', required=True, type=int,
                       help='number of epochs')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save the TensorBoard logs')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save the NumPy array consisting of the actual and predicted labels')
    arguments = parser.parse_args()
    return arguments


def main(arguments):

    # load the features of the dataset
    features = datasets.load_breast_cancer().data

    # standardize the features
    features = StandardScaler().fit_transform(features)

    # get the number of features
    num_features = features.shape[1]

    # load the corresponding labels for the features
    labels = datasets.load_breast_cancer().target

    # transform the labels to {-1, +1}
    labels[labels == 0] = -1

    # split the dataset to 70/30 partition: 70% train, 30% test
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.20, stratify=labels)

    train_size = train_features.shape[0]
    test_size = test_features.shape[0]

    # slice the dataset as per the batch size
    train_features = train_features[:train_size - (train_size % BATCH_SIZE)]
    train_labels = train_labels[:train_size - (train_size % BATCH_SIZE)]
    test_features = test_features[:test_size - (test_size % BATCH_SIZE)]
    test_labels = test_labels[:test_size - (test_size % BATCH_SIZE)]

    # instantiate the SVM class
    model = svm.SVM(alpha=LEARNING_RATE, batch_size=BATCH_SIZE, svm_c=arguments.svm_c, num_classes=NUM_CLASSES,
                    num_features=num_features)

    # train the instantiated model
    model.train(epochs=arguments.num_epochs, log_path=arguments.log_path, train_data=[train_features, train_labels],
                train_size=train_features.shape[0], validation_data=[test_features, test_labels],
                validation_size=test_features.shape[0], result_path=arguments.result_path)

    test_conf, test_accuracy = utils.plot_confusion_matrix(phase='testing', path=arguments.result_path,
                                                           class_names=['benign', 'malignant'])

    print('True negatives : {}'.format(test_conf[0][0]))
    print('False negatives : {}'.format(test_conf[1][0]))
    print('True positives : {}'.format(test_conf[0][1]))
    print('False positives : {}'.format(test_conf[1][1]))
    print('Testing accuracy : {}'.format(test_accuracy))


if __name__ == '__main__':
    args = parse_args()

    main(args)
