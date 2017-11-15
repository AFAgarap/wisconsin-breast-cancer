# A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and
# Support Vector Machine (SVM) for Intrusion Detection in Network Traffic Data
# Copyright (C) 2017  Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.1'
__author__ = 'Abien Fred Agarap'

from gru_svm import GruSvm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 256
CELL_SIZE = 256
DROPOUT_RATE = 0.8
LEARNING_RATE = 1e-3
NUM_CLASSES = 2
SVM_C = 1


def main():
    dataset = datasets.load_breast_cancer()

    features = dataset.data

    features = StandardScaler().fit_transform(features)

    num_features = features.shape[1]

    labels = dataset.target

    labels[labels == 0] = -1

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
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

    model = GruSvm(alpha=LEARNING_RATE, batch_size=BATCH_SIZE, cell_size=CELL_SIZE, dropout_rate=DROPOUT_RATE,
                   num_classes=NUM_CLASSES, sequence_length=num_features, svm_c=SVM_C)

    model.train(checkpoint_path='./checkpoint_path/', log_path='./log_path/', model_name='gru_svm', epochs=10000,
                train_data=[train_features, train_labels], train_size=train_size,
                validation_data=[test_features, test_labels],
                validation_size=test_size, result_path='./results')


if __name__ == '__main__':
    main()
