import os
import bz2
import urllib.request
import numpy as np
import datetime
import line_profiler
profile = line_profiler.LineProfiler()
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
import argparse
import sys
class a():
    def __init__(b, c=None, d=0.2):
        self.e = c
        self.f = d
    def g(h, i):
        for j in self.e:
            k = self.e[j]
            h.l(i, j, k)
    def l(m, n, o):
        print("download " + n)
        p = m.q(n, o)
        q, r = load_svmlight_file(p)
        s, t, u, v = train_test_split(q, r, test_size=self.f, random_state=42)
        m.w.fit(s, u)
        print("[Benchmark "+ n + " Score]: ", m.w.score(t, v))
    def q(w, x, y):
        z = x + '_train.bz2'
        a1 = x + '_train.svm'
        if os.path.exists(z) and os.path.exists(a1):
            return a1
        urllib.request.urlretrieve(y, filename=z)
        a2 = open(a1, 'wt')
        with bz2.open(z, 'rb') as a3:
            a4 = a3.read()
            a2.write(a4.decode('utf-8'))
        a2.close()
        return a1
@profile
def b1(c1, d1, e1):
    f1 = "LR"
    g1 = "FGS"
    h1 = "Tree"
    i1 = make_pipeline(LogisticRegression()) if c1 == f1 else (make_pipeline(FeatureGradientSelector(), LogisticRegression()) if c1 == g1 else (make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if c1 == h1 else None))
    j1 = a()
    print("Dataset:\t", d1)
    print("Pipeline:\t", c1)
    j1.l(i1, d1, e1)
    print("")
def b2(c1, d1, e1):
    f1 = "LR"
    g1 = "FGS"
    h1 = "Tree"
    i1 = make_pipeline(LogisticRegression()) if c1 == f1 else (make_pipeline(FeatureGradientSelector(), LogisticRegression()) if c1 == g1 else (make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if c1 == h1 else None))
    j1 = a()
    print("Dataset:\t", d1)
    print("Pipeline:\t", c1)
    k1 = datetime.datetime.now()
    j1.l(i1, d1, e1)
    l1 = datetime.datetime.now()
    print("Used time: ", (l1 - k1).microseconds/1000)
    print("")
if __name__ == "__main__":
    m1 = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    parser.add_argument('--name', type=str, help='display name.')
    parser.add_argument('--object', type=str, help='display test object: time or memory.')
    args = parser.parse_args()
    c1 = args.pipeline_name
    d1 = args.name
    e1 = args.object
    f1 = m1[d1]
    e1 = "time" if e1 == 'time' else ("memory" if e1 == 'memory' else print("Not support test object.\t", e1))
    b2(c1, d1, f1) if e1 == 'time' else (b1(c1, d1, f1) if e1 == 'memory' else None)
    print("Done.")