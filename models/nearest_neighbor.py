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

"""An implementation of K-Nearest Neighbor"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

dataset = datasets.load_breast_cancer()

features = dataset.data
labels = dataset.target

features = StandardScaler().fit_transform(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, stratify=labels)
train_labels = tf.one_hot(train_labels, depth=2, on_value=1, off_value=0)
test_labels = tf.one_hot(test_labels, depth=2, on_value=1, off_value=0)

xtr = tf.placeholder("float", [None, 30])
xte = tf.placeholder("float", [30])

distance = tf.sqrt(tf.reduce_sum(tf.square(xtr - xte), reduction_indices=1))

pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    y_, y = sess.run([test_labels, train_labels])

    # loop over test data
    for i in range(len(test_features)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: train_features, xte: test_features[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(y[nn_index]), "True Class:", np.argmax(y_[i]))
        # Calculate accuracy
        if np.argmax(y[nn_index]) == np.argmax(y_[i]):
            accuracy += 1./len(test_features)
    print("Done!")
print("Accuracy:", accuracy)