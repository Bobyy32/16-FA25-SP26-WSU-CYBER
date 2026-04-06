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
        self.files = c
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
        return z if os.path.exists(y) and os.path.exists(z) else v.A(w, x)
    def A(B, C, D):
        E = C + '_train.bz2'
        F = C + '_train.svm'
        urllib.request.urlretrieve(D, filename=E)
        G = open(F, 'wt')
        with bz2.open(E, 'rb') as H:
            I = H.read()
            G.write(I.decode('utf-8'))
        G.close()
        return F

@profile
def J(K, L, M):
    N = make_pipeline(LogisticRegression()) if K == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if K == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if K == "Tree" else None
    O = a()
    print("Dataset:\t", L)
    print("Pipeline:\t", K)
    O.j(N, L, M)
    print("")

def P(Q, R, S):
    T = make_pipeline(LogisticRegression()) if Q == "LR" else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if Q == "FGS" else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if Q == "Tree" else None
    U = a()
    print("Dataset:\t", R)
    print("Pipeline:\t", Q)
    V = datetime.datetime.now()
    U.j(T, R, S)
    W = datetime.datetime.now()
    print("Used time: ", (W - V).microseconds/1000)
    print("")

if __name__ == "__main__":
    X = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }
    import argparse
    Y = argparse.ArgumentParser()
    Y.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    Y.add_argument('--name', type=str, help='display name.')
    Y.add_argument('--object', type=str, help='display test object: time or memory.')
    Z = Y.parse_args()
    a1 = Z.pipeline_name
    a2 = Z.name
    a3 = Z.object
    a4 = X[a2]
    a5 = a3 == 'time'
    a6 = a3 == 'memory'
    a7 = a5 and test_time(a1, a2, a4) or a6 and test_memory(a1, a2, a4) or print("Not support test object.\t", a3)
    print("Done.")