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

import argparse

class a():
    def __init__(b, c=None, d=0.2):
        self.b = c
        self.d = d

    def e(f, g):
        for h in f.b:
            i = f.b[h]
            f.j(g, h, i)

    def j(k, l, m):
        print("download " + l)
        n = k.o(l, m)
        o, p = load_svmlight_file(n)
        q, r, s, t = train_test_split(o, p, test_size=k.d, random_state=42)
        g = l
        print("[Benchmark " + l + " Score]: ", g.score(r, t))

    def o(u, v, w):
        x = v + '_train.bz2'
        y = v + '_train.svm'
        z = os.path.exists(x) and os.path.exists(y)
        return y if z else u.a(v, w, x, y)

    def a(b, c, d, e):
        f = c + '_train.bz2'
        g = c + '_train.svm'
        if os.path.exists(f) and os.path.exists(g):
            return g
        urllib.request.urlretrieve(d, filename=f)
        h = open(g, 'wt')
        with bz2.open(f, 'rb') as i:
            j = i.read()
            h.write(j.decode('utf-8'))
        h.close()
        return g

def k(l, m, n):
    o = l == "LR" and make_pipeline(LogisticRegression()) or l == "FGS" and make_pipeline(FeatureGradientSelector(), LogisticRegression()) or l == "Tree" and make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
    p = a()
    print("Dataset:\t", m)
    print("Pipeline:\t", l)
    p.j(o, m, n)
    print("")

def q(r, s, t):
    u = r == "LR" and make_pipeline(LogisticRegression()) or r == "FGS" and make_pipeline(FeatureGradientSelector(), LogisticRegression()) or r == "Tree" and make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression())
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    parser.add_argument('--name', type=str, help='display name.')
    parser.add_argument('--object', type=str, help='display test object: time or memory.')
    args = parser.parse_args()
    z = args.pipeline_name
    aa = args.name
    ab = args.object
    ac = y[aa]
    ad = ab == 'time' and q(z, aa, ac) or ab == 'memory' and k(z, aa, ac) or print("Not support test object.\t", ab)
    print("Done.")