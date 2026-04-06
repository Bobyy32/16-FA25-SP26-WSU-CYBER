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

class a():
    def __init__(b, c=None, d=0.2):
        self.files =  c
        self.test_size = d

    def e(f, g):
        for h in f.files:
            i = f.files[h]
            f.j(g, h, i)

    def j(k, l, m):
        print("download " + l)
        n = k.o(l, m)
        p, q = load_svmlight_file(n)
        r, s, t, u = train_test_split(p, q, test_size=k.test_size, random_state=42)
        g.fit(r, t)
        print("[Benchmark "+ l + " Score]: ", g.score(s, u))

    def o(v, w, x):
        y = w + '_train.bz2'
        z = w + '_train.svm'
        return z if os.path.exists(y) and os.path.exists(z) else v.p(y, z, w, x)

    def p(q, r, s, t):
        urllib.request.urlretrieve(t, filename=q)
        u = open(r, 'wt')
        with bz2.open(q, 'rb') as v:
            w = v.read()
            u.write(w.decode('utf-8'))
        u.close()
        return r

@profile
def f(a, b, c):
    d = make_pipeline(LogisticRegression()) if a == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if a == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if a == "Tree" else None
    e = a()
    print("Dataset:\t", b)
    print("Pipeline:\t", a)
    e.j(d, b, c)
    print("")

def g(a, b, c):
    d = make_pipeline(LogisticRegression()) if a == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if a == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if a == "Tree" else None
    e = a()
    print("Dataset:\t", b)
    print("Pipeline:\t", a)
    f = datetime.datetime.now()
    e.j(d, b, c)
    g = datetime.datetime.now()
    print("Used time: ", (g - f).microseconds/1000)
    print("")

if __name__ == "__main__":
    h = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }

    import argparse
    i = argparse.ArgumentParser()
    i.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    i.add_argument('--name', type=str, help='display name.')
    i.add_argument('--object', type=str, help='display test object: time or memory.')
    j = i.parse_args()
    k = j.pipeline_name
    l = j.name
    m = j.object
    n = h[l]
    o = test_time(k, l, n) if m == 'time' else test_memory(k, l, n) if m == 'memory' else print("Not support test object.\t", m)
    print("Done.")