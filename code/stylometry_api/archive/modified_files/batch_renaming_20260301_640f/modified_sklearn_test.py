import numpy as np
import urllib.request
import bz2

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier

from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector


def a():
    c = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
    urllib.request.urlretrieve(c, filename='b.bz2')

    d = open('b.svm', 'wt')
    with bz2.open('b.bz2', 'rb') as e:
        f = e.read()
        d.write(f.decode('utf-8'))
    d.close()

    g, h = load_svmlight_file('b.svm')
    i, j, k, l = train_test_split(g, h, test_size=0.33, random_state=42)

    m = make_pipeline(FeatureGradientSelector(n_epochs=1, n_features=10), LogisticRegression())
    m.fit(i, k)

    print("Pipeline Score: ", m.score(i, k))

if __name__ == "__main__":
    a()