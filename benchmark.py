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

"""Implementation of classifier algorithms for the Wisconsin breast cancer dataset"""
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

__version__ = '0.1'
__author__ = 'Abien Fred Agarap'

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

breast_cancer_data = datasets.load_breast_cancer()

x = breast_cancer_data.data
y = breast_cancer_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y)

svm_c = [1, 10, 100]
loss = ['hinge', 'squared_hinge']
svm_result = []

for l in loss:
    for c in svm_c:
        model = LinearSVC(loss=l, C=c)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        result = 'SVM, Loss={} C={}\nAccuracy : {}\n'.format(l, c, accuracy)
        print(result)
        svm_result.extend(result)

for result in svm_result:
    with open('svm.txt', 'a') as file:
        file.write(result)

kernel = ['rbf', 'linear', 'sigmoid', 'poly']
degree = [1, 2, 3, 4, 5]
svc_result = []

for k in kernel:
    for c in svm_c:
        model = SVC(C=c, kernel=k)
        model.fit(x_train, y_train)
        result = 'SVC : C={} K={} Accuracy : {}\n'.format(c, k, model.score(x_test, y_test))
        print(result)
        svc_result.extend(result)

for result in svc_result:
    with open('svc.txt', 'a') as file:
        file.write(result)
