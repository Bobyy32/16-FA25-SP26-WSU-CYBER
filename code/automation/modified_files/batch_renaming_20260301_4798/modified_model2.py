import torch
import torch.nn as nn
import torch.nn.functional as F
import ops
import numpy as np
from nni.nas.pytorch import mutables
from utils import parse_results
from aux_head import *

class a(nn.Module):
    def __init__(b, c, d, e, f):
        super().__init__()
        self.g = nn.ModuleList()
        h = []
        for i in range(d):
            j = 2 if i < f else 1
            h.append("{}_p{}".format(c, i))
            self.g.append(mutables.LayerChoice([ops.OPS[k](e, j, False) for k in ops.PRIMITIVES],
                                                key=h[-1]))
        self.k = ops.DropPath()
        self.l = mutables.InputChoice(choose_from=h, n_chosen=2, key="{}_switch".format(c))

    def forward(m, n):
        assert len(self.g) == len(n)
        o = [p(q) for p, q in zip(self.g, n)]
        o = [self.k(r) if r is not None else None for r in o]
        return self.l(o)


class p(nn.Module):
    def __init__(q, r, s, t, u, v, w):
        super().__init__()
        self.x = w
        self.y = r

        if v:
            self.z = ops.FactorizedReduce(s, u, affine=False)
        else:
            self.z = ops.StdConv(s, u, 1, 1, 0, affine=False)
        self.a1 = ops.StdConv(t, u, 1, 1, 0, affine=False)

        self.a2 = nn.ModuleList()
        for a3 in range(2, self.y + 2):
            self.a2.append(a("{}_n{}".format("reduce" if w else "normal", a3),
                             a3, u, 2 if w else 0))

    def forward(a4, a5, a6):
        a7 = [self.z(a5), self.a1(a6)]
        for a8 in self.a2:
            a9 = a8(a7)
            a7.append(a9)

        a10 = torch.cat(a7[2:], dim=1)
        return a10


class a11(nn.Module):
    def __init__(a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22):
        super().__init__()
        assert a13 in ["cifar10", "imagenet"]
        self.a23 = a13
        self.a24 = 32 if a13 == "cifar" else 224
        self.a25 = a15
        self.a26 = a16
        self.a27 = a17
        self.a28 = a22
        self.a29 = {2 * a14 // 3: self.a24 // 4}
        if a13 == "cifar10":
            self.a30 = 10
            self.a31 = AuxiliaryHeadCIFAR if a21 else DistillHeadCIFAR
            if not a21:
                self.a29 = {a14 // 3: 6, 2 * a14 // 3: 6}
        elif a13 == "imagenet":
            self.a30 = 1000
            self.a31 = AuxiliaryHeadImageNet if a21 else DistillHeadImagenet
            if not a21:
                self.a29 = {a14 // 3: 6, 2 * a14 // 3: 5}
        self.a32 = a14
        self.a33 = nn.ModuleDict()
        self.a34 = nn.Parameter(torch.rand(len(self.a29) + 1) / (len(self.a29) + 1)) \
            if not a21 else None

        a35 = 3 if a13 == "cifar" else 1
        a36 = a35 * self.a26
        self.a37 = {}
        if a20 is not None:
            self.a38 = a20["stem"]
        else:
            self.a38 = nn.Sequential(
                nn.Conv2d(a15, a36, 3, 1, 1, bias=False),
                nn.BatchNorm2d(a36)
            )
            self.a37["stem"] = self.a38

        a39, a40, a36 = a36, a36, self.a26

        self.a41 = nn.ModuleList()
        a42, a43 = False, False
        a44 = 0
        for a45 in range(a14):
            a42, a43 = a43, False
            if a45 in [a14 // 3, 2 * a14 // 3]:
                a36 *= 2
                a43 = True

            a46 = p(a17, a39, a40, a36, a42, a43)
            self.a41.append(a46)
            a47 = a36 * a17
            if a45 in self.a29:
                if a20 is not None:
                    self.a33[str(a45)] = a20["aux" + str(a44)]
                else:
                    self.a33[str(a45)] = self.a31(a47, self.a29[a45], self.a30)
                    self.a37["aux" + str(a44)] = self.a33[str(a45)]
                a44 += 1
            a39, a40 = a40, a47

        self.a48 = nn.AdaptiveAvgPool2d(1)
        self.a49 = nn.Linear(a40, self.a30)

    def forward(a50, a51):
        a52 = a53 = self.a38(a51)
        a54 = []

        for a55, a56 in enumerate(self.a41):
            a52, a53 = a53, a56(a52, a53)
            if str(a55) in self.a33:
                a54.append(self.a33[str(a55)](a53))

        a57 = self.a48(a53)
        a57 = a57.view(a57.size(0), -1)
        a58 = self.a49(a57)
        a54.append(a58)

        if self.a34 is None:
            assert len(a54) == 2
            return a54[1], a54[0]
        else:
            a59 = torch.cat([(a60 * a61) for a60, a61 in zip(F.softmax(self.a34, dim=0), a54)], 0)
            return a58, a59

    def a62(self, a63):
        for a64 in self.modules():
            if isinstance(a64, ops.DropPath):
                a64.p = a63

    def a65(self, a66, a67):
        a68 = parse_results(a66, self.a27)
        a67.info(a68)
        return a68