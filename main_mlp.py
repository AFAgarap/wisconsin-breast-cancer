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

"""Main program implementing the MLP class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

import argparse
from models.MLP import MLP
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BATCH_SIZE = 128
LEARNING_RATE = 1e-2
NUM_CLASSES = 2
NUM_NODES = [500, 500, 500]


def parse_args():
    parser = argparse.ArgumentParser(
        description='MLP written using TensorFlow, for Wisconsin Breast Cancer Diagnostic Dataset')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-n', '--num_epochs', required=True, type=int,
                       help='number of epochs')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='path where to save the TensorBoard logs')
    group.add_argument('-r', '--result_path', required=True, type=str,
                       help='path where to save actual and predicted labels array')
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    # load the features of the dataset
    features = datasets.load_breast_cancer().data

    # standardize the features
    features = StandardScaler().fit_transform(features)

    # get the number of features
    num_features = features.shape[1]

    # load the labels for the features
    labels = datasets.load_breast_cancer().target

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.30,
                                                                                stratify=labels)

    model = MLP(alpha=LEARNING_RATE, batch_size=BATCH_SIZE, node_size=NUM_NODES, num_classes=NUM_CLASSES,
                num_features=num_features)

    model.train(num_epochs=arguments.num_epochs, log_path=arguments.log_path, train_data=[train_features, train_labels],
                train_size=train_features.shape[0], test_data=[test_features, test_labels],
                test_size=test_features.shape[0], result_path=arguments.result_path)


if __name__ == '__main__':
    args = parse_args()

    main(args)
