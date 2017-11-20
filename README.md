Using Machine Learning Algorithms for the Detection of Breast Cancer using the Wisconsin Diagnostic Dataset
===

[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)]()
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)]()

## Abstract
This paper presents a comparison of six machine learning (ML) algorithms: <a href="https://github.com/AFAgarap/gru-svm">
GRU-SVM</a><a href="http://arxiv.org/abs/1709.03082">[4]</a>, Linear Regression, Multilayer Perceptron (MLP),
Nearest Neighbor (NN) search, Softmax Regression, and Support Vector Machine (SVM) on the Wisconsin Diagnostic Breast
Cancer (WDBC) dataset <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)">[22]</a>
by measuring their classification test accuracy and their sensitivity and specificity values. The said dataset consists
of features which were computed from digitized images of FNA tests on a breast mass[22]. For the implementation of
the ML algorithms, the dataset was partitioned in the following fashion: 70% for training phase, and 30% for the
testing phase. The hyper-parameters used for all the classifiers were manually assigned. Results show that all the
presented ML algorithms performed well (all exceeded 90% test accuracy) on the classification task. The MLP algorithm
stands out among the implemented algorithms with a test accuracy of ~99.04% Lastly, the results are comparable
with the findings of the related studies[<a href="https://www.ijcit.com/archives/volume1/issue1/Paper010105.pdf">18</a>
, <a href="https://link.springer.com/chapter/10.1007%2F0-387-34224-9_58?LI=true">23</a>].

## Machine Learning (ML) Algorithms

* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_gru_svm.py">GRU-SVM</a>
* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_linear_regression.py">Linear Regression</a>
* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_mlp.py">Multilayer Perceptron</a>
* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_nearest_neighbor.py">Nearest Neighbor</a>
* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_logistic_regression.py">Softmax Regression</a>
* <a href="https://github.com/AFAgarap/wisconsin-breast-cancer/blob/master/main_svm.py">L2-SVM</a> 

## License
```buildoutcfg
Copyright 2017 Abien Fred Agarap

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
