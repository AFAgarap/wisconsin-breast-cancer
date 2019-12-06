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

"""An implementation of K-Nearest Neighbor class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

from models.nearest_neighbor import NearestNeighbor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    dataset = datasets.load_breast_cancer()

    features = dataset.data
    labels = dataset.target

    num_features = features.shape[1]

    features = StandardScaler().fit_transform(features)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, stratify=labels
    )

    model = NearestNeighbor(train_features, train_labels, num_features)

    model.predict(test_features, test_labels, result_path="./results/nearest_neighbor/")


if __name__ == "__main__":
    main()
