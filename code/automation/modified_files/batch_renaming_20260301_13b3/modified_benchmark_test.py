import os
import bz2
import urllib.request
import numpy as np
import datetime
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
        g = k.fit(r, t)
        print("[Benchmark "+ l + " Score]: ", g.score(s, u))

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

def f(h, i, j):
    k = make_pipeline(LogisticRegression()) if h == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if h == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if h == "Tree" else None
    l = a()
    print("Dataset:\t", i)
    print("Pipeline:\t", h)
    l.j(k, i, j)
    print("")

def m(n, o, p):
    q = make_pipeline(LogisticRegression()) if n == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if n == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if n == "Tree" else None
    r = a()
    print("Dataset:\t", o)
    print("Pipeline:\t", n)
    s = datetime.datetime.now()
    r.j(q, o, p)
    t = datetime.datetime.now()
    print("Used time: ", (t - s).microseconds/1000)
    print("")

if __name__ == "__main__":
    u = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }
    import argparse
    v = argparse.ArgumentParser()
    v.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    v.add_argument('--name', type=str, help='display name.')
    v.add_argument('--object', type=str, help='display test object: time or memory.')
    w = v.parse_args()
    x = w.pipeline_name
    y = w.name
    z = w.object
    A = u[y]
    B = "time"
    C = "memory"
    D = "Not support test object.\t"
    print("Done." if z == C else print(D + z) if z != B else test_time(x, y, A) if z == B else test_memory(x, y, A))