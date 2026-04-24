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
        self.a28 = {2 * a14 // 3: self.a24 // 4}
        if a13 == "cifar10":
            self.a29 = 10
            self.a30 = AuxiliaryHeadCIFAR if a21 else DistillHeadCIFAR
            if not a21:
                self.a28 = {a14 // 3: 6, 2 * a14 // 3: 6}
        elif a13 == "imagenet":
            self.a29 = 1000
            self.a30 = AuxiliaryHeadImageNet if a21 else DistillHeadImagenet
            if not a21:
                self.a28 = {a14 // 3: 6, 2 * a14 // 3: 5}
        self.a31 = a14
        self.a32 = nn.ModuleDict()
        self.a33 = nn.Parameter(torch.rand(len(self.a28) + 1) / (len(self.a28) + 1)) \
            if not a21 else None

        a34 = 3 if a13 == "cifar" else 1
        a35 = a34 * self.a26
        self.a36 = {}
        if a22 is not None:
            self.a37 = a22["stem"]
        else:
            self.a37 = nn.Sequential(
                nn.Conv2d(a15, a35, 3, 1, 1, bias=False),
                nn.BatchNorm2d(a35)
            )
            self.a36["stem"] = self.a37

        a38, a39, a35 = a35, a35, self.a26

        self.a40 = nn.ModuleList()
        a41, a42 = False, False
        a43 = 0
        for a44 in range(a14):
            a41, a42 = a42, False
            if a44 in [a14 // 3, 2 * a14 // 3]:
                a35 *= 2
                a42 = True

            a45 = p(a17, a38, a39, a35, a41, a42)
            self.a40.append(a45)
            a46 = a35 * a17
            if a44 in self.a28:
                if a22 is not None:
                    self.a32[str(a44)] = a22["aux" + str(a43)]
                else:
                    self.a32[str(a44)] = self.a30(a46, self.a28[a44], self.a29)
                    self.a36["aux" + str(a43)] = self.a32[str(a44)]
                a43 += 1
            a38, a39 = a39, a46

        self.a47 = nn.AdaptiveAvgPool2d(1)
        self.a48 = nn.Linear(a39, self.a29)

    def forward(a49, a50):
        a51 = a52 = self.a37(a50)
        a53 = []

        for a54, a55 in enumerate(self.a40):
            a51, a52 = a52, a55(a51, a52)
            if str(a54) in self.a32:
                a53.append(self.a32[str(a54)](a52))

        a56 = self.a47(a52)
        a56 = a56.view(a56.size(0), -1)
        a57 = self.a48(a56)
        a53.append(a57)

        if self.a33 is None:
            assert len(a53) == 2
            return a53[1], a53[0]
        else:
            a58 = torch.cat([(a59 * a60) for a59, a60 in zip(F.softmax(self.a33, dim=0), a53)], 0)
            return a57, a58

    def a61(self, a62):
        for a63 in self.modules():
            if isinstance(a63, ops.DropPath):
                a63.p = a62

    def a64(self, a65, a66):
        a67 = parse_results(a65, self.a27)
        a66.info(a67)
        return a67