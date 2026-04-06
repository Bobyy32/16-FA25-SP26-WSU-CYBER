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
        print("[Benchmark "+ l + " Score]: ", g.score(s, u)

    def o(v, w, x):
        y = w + '_train.bz2'
        z = w + '_train.svm'
        return z if os.path.exists(y) and os.path.exists(z) else v.a(y, x, z)

    def a(b, c, d):
        urllib.request.urlretrieve(c, filename=b)
        e = open(d, 'wt')
        with bz2.open(b, 'rb') as f:
            g = f.read()
            e.write(g.decode('utf-8'))
        e.close()
        return d

@profile
def h(i, j, k):
    l = make_pipeline(LogisticRegression()) if i == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if i == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if i == "Tree" else None
    m = a()
    print("Dataset:\t", j)
    print("Pipeline:\t", i)
    m.j(l, j, k)
    print("")

def f(i, j, k):
    l = make_pipeline(LogisticRegression()) if i == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if i == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if i == "Tree" else None
    m = a()
    print("Dataset:\t", j)
    print("Pipeline:\t", i)
    n = datetime.datetime.now()
    m.j(l, j, k)
    o = datetime.datetime.now()
    print("Used time: ", (o - n).microseconds/1000)
    print("")

if __name__ == "__main__":
    p = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }

    import argparse
    q = argparse.ArgumentParser()
    q.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    q.add_argument('--name', type=str, help='display name.')
    q.add_argument('--object', type=str, help='display test object: time or memory.')
    r = q.parse_args()
    s = r.pipeline_name
    t = r.name
    u = r.object
    v = p[t]
    w = test_time(s, t, v) if u == 'time' else test_memory(s, t, v) if u == 'memory' else print("Not support test object.\t", u)
    print("Done.")