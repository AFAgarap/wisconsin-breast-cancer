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

"""Implementation of K-fold Cross Validation for ML Algorithms"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'Abien Fred Agarap'
__version__ = '0.1.0'

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def main():
    dataset = datasets.load_breast_cancer()
    features = dataset.data
    labels = dataset.target
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, stratify=labels)
    
    parameter_set = {'loss': ('hinge', 'squared_hinge'), 'C': [1, 10, 100, 1000, 5, 50, 500, 5000]}

    model = LinearSVC()
    
    clf = GridSearchCV(estimator=model, param_grid=parameter_set, n_jobs=3, cv=10)

    clf.fit(train_features, train_labels)

    print(clf.grid_scores_)
    print('clf.best_score_ : {}'.format(clf.best_score_))
    print('clf.best_params_ : {}'.format(clf.best_params_))
    print('clf.score : {}'.format(clf.score(test_features, test_labels)))


if __name__ == '__main__':
    main()
