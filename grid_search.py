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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def validate_model(model, parameter_set, train_data, test_data):
    clf = GridSearchCV(estimator=model, param_grid=parameter_set, n_jobs=3, cv=10)

    clf.fit(train_data[0], train_data[1])

    grid_scores = clf.grid_scores_
    best_score = clf.best_score_
    best_params = clf.best_params_
    test_score = clf.score(test_data[0], test_data[1])
    
    return grid_scores, best_score, best_params, test_score


def main():
    dataset = datasets.load_breast_cancer()
    features = dataset.data
    labels = dataset.target
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
        stratify=labels)
    
    parameter_set = {'loss': ('hinge', 'squared_hinge'), 'C': [1, 10, 100, 1000, 5, 50, 500, 5000]}

    model = LinearSVC()
    grid_scores, best_score, best_params, test_score = validate_model(model=model, parameter_set=parameter_set,
        train_data=[train_features, train_labels], test_data=[test_features, test_labels])

    print(grid_scores)
    print('SVM best score: {}'.format(best_score))
    print('SVM best params : {}'.format(best_params))
    print('SVM test score : {}'.format(test_score))

    parameter_set = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'batch_size': [16, 32, 64, 128],}

    model = MLPClassifier()

    grid_scores, best_score, best_params, test_score = validate_model(model=model, parameter_set=parameter_set,
        train_data=[train_features, train_labels], test_data=[test_features, test_labels])

    print(grid_scores)
    print('MLP best score: {}'.format(best_score))
    print('MLP best params : {}'.format(best_params))
    print('MLP test score : {}'.format(test_score))


if __name__ == '__main__':
    main()
