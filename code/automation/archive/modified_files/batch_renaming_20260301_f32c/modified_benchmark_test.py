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
        return z if os.path.exists(y) and os.path.exists(z) else v.a(w, x, y, z)

    def a(b, c, d, e):
        f = c + '_train.bz2'
        g = c + '_train.svm'
        urllib.request.urlretrieve(d, filename=f)
        h = open(g, 'wt')
        with bz2.open(f, 'rb') as i:
            j = i.read()
            h.write(j.decode('utf-8'))
        h.close()
        return g

@profile
def k(l, m, n):
    o = make_pipeline(LogisticRegression()) if l == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if l == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
    p = a()
    print("Dataset:\t", m)
    print("Pipeline:\t", l)
    p.j(o, m, n)
    print("")

def q(r, s, t):
    u = make_pipeline(LogisticRegression()) if r == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if r == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
    v = a()
    print("Dataset:\t", s)
    print("Pipeline:\t", r)
    w = datetime.datetime.now()
    v.j(u, s, t)
    x = datetime.datetime.now()
    print("Used time: ", (x - w).microseconds/1000)
    print("")

if __name__ == "__main__":
    y = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }

    import argparse
    z = argparse.ArgumentParser()
    z.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    z.add_argument('--name', type=str, help='display name.')
    z.add_argument('--object', type=str, help='display test object: time or memory.')
    A = z.parse_args()
    B = A.pipeline_name
    C = A.name
    D = A.object
    E = y[C]
    F = test_time(B, C, E) if D == 'time' else test_memory(B, C, E) if D == 'memory' else print("Not support test object.\t", D)
    print("Done.")