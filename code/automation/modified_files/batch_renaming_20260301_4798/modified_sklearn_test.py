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
    i = h.decode('utf-8')
    e.write(i)
    e.close()
    j = load_svmlight_file(d)
    k = train_test_split(j[0], j[1], test_size=0.33, random_state=42)
    l = make_pipeline(FeatureGradientSelector(n_epochs=1, n_features=10), LogisticRegression())
    m = l.fit(k[0], k[2])
    n = l.score(k[0], k[2])
    print("Pipeline Score: ", n)

if __name__ == "__main__":
    a()