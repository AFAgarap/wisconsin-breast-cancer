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

"""Implements the Logistic Regression class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from models.linear_regression import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = 2


def main():
    dataset = datasets.load_breast_cancer()

    features = dataset.data

    features = StandardScaler().fit_transform(features)

    num_features = features.shape[1]

    labels = dataset.target

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                stratify=labels)

    train_size = train_features.shape[0]
    test_size = test_features.shape[0]

    # slice the dataset to be exact as per the batch size
    # e.g. train_size = 1898322, batch_size = 256
    # [:1898322-(1898322%256)] = [:1898240]
    # 1898322 // 256 = 7415; 7415 * 256 = 1898240
    train_features = train_features[:train_size - (train_size % BATCH_SIZE)]
    train_labels = train_labels[:train_size - (train_size % BATCH_SIZE)]

    # modify the size of the dataset to be passed on model.train()
    train_size = train_features.shape[0]

    # slice the dataset to be exact as per the batch size
    test_features = test_features[:test_size - (test_size % BATCH_SIZE)]
    test_labels = test_labels[:test_size - (test_size % BATCH_SIZE)]

    test_size = test_features.shape[0]

    model = LinearRegression(alpha=LEARNING_RATE, batch_size=BATCH_SIZE, num_classes=NUM_CLASSES,
                             sequence_length=num_features)

    model.train(epochs=2500, log_path='./log_path/linear_regression', train_data=[train_features, train_labels],
                train_size=train_size, validation_data=[test_features, test_labels], validation_size=test_size,
                result_path='./results/linear_regression')


if __name__ == '__main__':
    main()
