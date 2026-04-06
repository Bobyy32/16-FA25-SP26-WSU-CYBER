# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import bz2
import urllib.request
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector


def run_experiment():
    train_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
    urllib.request.urlretrieve(train_url, filename='data.bz2')

    svm_file = open('data.svm', 'wt')
    with bz2.open('data.bz2', 'rb') as zip_file:
        content = zip_file.read()
        svm_file.write(content.decode('utf-8'))
    svm_file.close()


    features, labels = load_svmlight_file('data.svm')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)


    model_pipeline = make_pipeline(FeatureGradientSelector(n_epochs=1, n_features=10), LogisticRegression())
    # model_pipeline = make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())

    model_pipeline.fit(x_train, y_train)

    print("Model Score: ", model_pipeline.score(x_train, y_train))

if __name__ == "__main__":
    run_experiment()