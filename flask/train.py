# Copyright 2018 Abien Fred Agarap
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

"""Keras implementation of DNN for Flask app"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0."
__author__ = "Abien Fred Agarap https://AFAgarap.me"

import argparse
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="MLP for Breast Cancer Detection")
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-d", "--dataset", required=True, type=str, help="the WDBC dataset"
    )
    arguments = parser.parse_args()
    return arguments


def main(args):

    # get the dataset filename from parse_args()
    dataset = args.dataset

    # names for the dataset features
    column_names = [
        "Sample code number",
        "Clump Thickness",
        "Uniformity of Cell Size",
        "Uniformity of Cell Shape",
        "Marginal Adhesion",
        "Single Epithelial Cell Size",
        "Bare Nuclei",
        "Bland Chromatin",
        "Normal Nucleoli",
        "Mitoses",
        "Class",
    ]

    # load the dataset
    data = pd.read_csv(dataset, names=column_names, delimiter=",")

    # replace missing data with NaN
    data = data.replace("?", np.NaN)

    # replace NaN data with 0
    data = data.fillna(0)

    # get the dataset features
    features = np.array(data.iloc[:, 0:10], np.int)

    # get the feature labels
    labels = np.array(data.iloc[:, 10], np.int)

    # convert the benign label to 0
    labels[labels == 2] = 0

    # convert the malignant label to 1
    labels[labels == 4] = 1

    # split the dataset into 70/30 for training and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    # standardize using StandardScaler
    scaler = StandardScaler()

    # fit the scaler to train features
    scaler.fit(x_train)

    # standardize the features
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # define the keras DNN model
    model = Sequential()

    # accept input with shape of features, use ReLU
    model.add(Dense(512, input_dim=features.shape[1], activation="relu"))

    # add two hidden layers with ReLU
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))

    # add dropout layer
    model.add(Dropout(0.25))

    # binary classifier using sigmoid
    model.add(Dense(1, activation="sigmoid"))

    # binary cross-entropy as loss function, and adam as optimizer
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train the model by 32 epochs
    model.fit(x_train, y_train, epochs=32, verbose=1)

    # evaluate the trained model
    accuracy = model.evaluate(x_test, y_test)[1]

    # display evaluation accuracy
    print("Test accuracy : {}".format(accuracy))

    # save the trained model
    model.save("dnn.h5")


if __name__ == "__main__":
    args = parse_args()

    main(args)
