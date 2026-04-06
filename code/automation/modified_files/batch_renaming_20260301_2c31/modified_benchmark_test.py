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
import line_profiler
profile = line_profiler.LineProfiler()
class a():
    def __init__(b, c=None, d=0.2):
        self.files =  c
        self.test_size = d
    def e(f, g):
        for h in self.files:
            i = self.files[h]
            self.j(g, h, i)
    def j(k, l, m):
        print("download " + l)
        n = self.o(l, m)
        p, q = load_svmlight_file(n)
        r, s, t, u = train_test_split(p, q, test_size=self.test_size, random_state=42)
        k.fit(r, t)
        print("[Benchmark "+ l + " Score]: ", k.score(s, u))
    def o(v, w):
        x = v + '_train.bz2'
        y = v + '_train.svm'
        return y if os.path.exists(x) and os.path.exists(y) else (urllib.request.urlretrieve(w, filename=x), f_svm := open(y, 'wt'), f_zip := bz2.open(x, 'rb'), data := f_zip.read(), f_svm.write(data.decode('utf-8')), f_svm.close(), y)[6]
@profile
def z(A, B, C):
    return make_pipeline(LogisticRegression()) if A == "LR" else (make_pipeline(FeatureGradientSelector(), LogisticRegression()) if A == "FGS" else (make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if A == "Tree" else None))
    D = a()
    print("Dataset:\t", B)
    print("Pipeline:\t", A)
    D.j(z(A, B, C), B, C)
    print("")
def E(A, B, C):
    return make_pipeline(LogisticRegression()) if A == "LR" else (make_pipeline(FeatureGradientSelector(), LogisticRegression()) if A == "FGS" else (make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if A == "Tree" else None))
    D = a()
    print("Dataset:\t", B)
    print("Pipeline:\t", A)
    F = datetime.datetime.now()
    D.j(E(A, B, C), B, C)
    G = datetime.datetime.now()
    print("Used time: ", (G - F).microseconds/1000)
    print("")
if __name__ == "__main__":
    H = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }
    import argparse
    I = argparse.ArgumentParser()
    I.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    I.add_argument('--name', type=str, help='display name.')
    I.add_argument('--object', type=str, help='display test object: time or memory.')
    J = I.parse_args()
    K = J.pipeline_name
    L = J.name
    M = J.object
    N = H[L]
    print("Not support test object.\t", M) if M != 'time' and M != 'memory' else (E(K, L, N) if M == 'time' else z(K, L, N))
    print("Done.")