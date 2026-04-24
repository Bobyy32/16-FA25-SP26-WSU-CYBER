# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from nni.algorithms.feature_engineering.gradient_selector import FeatureGradientSelector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import os
import datetime
import numpy as np
import urllib.request
import bz2

import line_profiler
profile = line_profiler.LineProfiler()

class A():
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
        return y if os.path.exists(x) and os.path.exists(y) else self.z(v, w, x, y)

    def z(a, b, c, d):
        urllib.request.urlretrieve(b, filename=c)
        e = open(d, 'wt')
        with bz2.open(c, 'rb') as f:
            g = f.read()
            e.write(g.decode('utf-8'))
        e.close()
        return d

@profile
def h(i, j, k):
    l = "LR"
    m = "FGS"
    n = "Tree"
    o = make_pipeline(LogisticRegression()) if i == l else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if i == m else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if i == n else None
    p = A()
    print("Dataset:\t", j)
    print("Pipeline:\t", i)
    p.j(o, j, k)
    print("")

def f(i, j, k):
    l = "LR"
    m = "FGS"
    n = "Tree"
    o = make_pipeline(LogisticRegression()) if i == l else make_pipeline(FeatureGradientSelector(), LogisticRegression()) if i == m else make_pipeline(SelectFromModel(ExtraTreesClassifier(n_estimators=50)), LogisticRegression()) if i == n else None
    p = A()
    print("Dataset:\t", j)
    print("Pipeline:\t", i)
    q = datetime.datetime.now()
    p.j(o, j, k)
    r = datetime.datetime.now()
    print("Used time: ", (r - q).microseconds/1000)
    print("")

if __name__ == "__main__":
    s = {
        "rcv1" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "colon-cancer" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "gisette" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "news20.binary" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "real-sim" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "webspam" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
        "avazu" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2"
    }
    import argparse
    t = argparse.ArgumentParser()
    t.add_argument('--pipeline_name', type=str, help='display pipeline_name.')
    t.add_argument('--name', type=str, help='display name.')
    t.add_argument('--object', type=str, help='display test object: time or memory.')
    u = t.parse_args()
    v = u.pipeline_name
    w = u.name
    x = u.object
    y = s[w]
    z = 'time'
    A = 'memory'
    test_time(v, w, y) if x == z else test_memory(v, w, y) if x == A else print("Not support test object.\t", x)
    print("Done.")