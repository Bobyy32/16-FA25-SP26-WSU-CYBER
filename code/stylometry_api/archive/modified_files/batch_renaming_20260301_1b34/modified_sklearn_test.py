import random
import sys
import os
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector

def a():
    b = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
    c = 'train.bz2'
    d = 'train.svm'
    e = open(d, 'wt')
    f = open(c, 'wb')
    f.write(bz2.decompress(open(c, 'rb').read()))
    f.close()
    g = open(c, 'rb')
    h = g.read()
    e.write(h.decode('utf-8'))
    e.close()
    i, j = load_svmlight_file(d)
    k, l, m, n = train_test_split(i, j, test_size=0.33, random_state=42)
    o = make_pipeline(FeatureGradientSelector(n_epochs=1, n_features=10), LogisticRegression())
    o.fit(k, m)
    print("Pipeline Score: ", o.score(k, m))

if __name__ == "__main__":
    a()