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
        u = s + '_train.bz2'
        v = s + '_train.svm'
        urllib.request.urlretrieve(t, filename=u)
        w = open(v, 'wt')
        with bz2.open(u, 'rb') as x:
            y = x.read()
            w.write(y.decode('utf-8'))
        w.close()
        return v

@profile
def f(g, h, i):
    j = make_pipeline(LogisticRegression()) if g == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if g == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if g == "Tree" else None
    k = a()
    print("Dataset:\t", h)
    print("Pipeline:\t", g)
    k.j(j, h, i)
    print("")

def l(m, n, o):
    p = make_pipeline(LogisticRegression()) if m == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if m == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if m == "Tree" else None
    q = a()
    print("Dataset:\t", n)
    print("Pipeline:\t", m)
    r = datetime.datetime.now()
    q.j(p, n, o)
    s = datetime.datetime.now()
    print("Used time: ", (s - r).microseconds/1000)
    print("")

if __name__ == "__main__":
    t = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }

    import argparse
    u = argparse.ArgumentParser()
    u.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    u.add_argument('--name', type=str, help='display name.')
    u.add_argument('--object', type=str, help='display test object: time or memory.')
    v = u.parse_args()
    w = v.pipeline_name
    x = v.name
    y = v.object
    z = t[x]
    test_object = y
    pipeline_name = w
    name = x
    path = z
    test_benchmark = a()
    test_benchmark.run_all_test(None)
    print("Done.")