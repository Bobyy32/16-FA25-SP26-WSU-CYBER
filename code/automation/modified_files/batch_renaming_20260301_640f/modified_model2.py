# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from utils import parse_results
from aux_head import DistillHeadCIFAR, DistillHeadImagenet, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops
import numpy as np
from nni.nas.pytorch import mutables


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

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if v:
            self.z = ops.FactorizedReduce(s, u, affine=False)
        else:
            self.z = ops.StdConv(s, u, 1, 1, 0, affine=False)
        self.a1 = ops.StdConv(t, u, 1, 1, 0, affine=False)

        # generate dag
        self.a2 = nn.ModuleList()
        for a3 in range(2, self.y + 2):
            self.a2.append(a("{}_n{}".format("reduce" if w else "normal", a3),
                                         a3, u, 2 if w else 0))

    def forward(a4, a5, a6):
        # a5, a6 are the outputs of previous previous cell and previous cell, respectively.
        a7 = [self.z(a5), self.a1(a6)]
        for a8 in self.a2:
            a9 = a8(a7)
            a7.append(a9)

        a10 = torch.cat(a7[2:], dim=1)
        return a10


class a11(nn.Module):

    def __init__(a12, a13, a14, a15=3, a16=16, a17=4, a18=False, a19=None):
        super().__init__()
        assert a13 in ["cifar10", "imagenet"]
        self.a20 = a13
        self.a21 = 32 if a13 == "cifar" else 224
        self.a22 = a15
        self.a23 = a16
        self.a24 = a17
        self.a25 = {2 * a14 // 3: self.a21 // 4}
        if a13 == "cifar10":
            self.a26 = 10
            self.a27 = AuxiliaryHeadCIFAR if a18 else DistillHeadCIFAR
            if not a18:
                self.a25 = {a14 // 3: 6, 2 * a14 // 3: 6}
        elif a13 == "imagenet":
            self.a26 = 1000
            self.a27 = AuxiliaryHeadImageNet if a18 else DistillHeadImagenet
            if not a18:
                self.a25 = {a14 // 3: 6, 2 * a14 // 3: 5}
        self.a28 = a14
        self.a29 = nn.ModuleDict()
        self.a30 = nn.Parameter(torch.rand(len(self.a25) + 1) / (len(self.a25) + 1)) \
            if not a18 else None

        a31 = 3 if a13 == "cifar" else 1
        a32 = a31 * self.a23
        self.a33 = {}
        if a19 is not None:
            self.a34 = a19["stem"]
        else:
            self.a34 = nn.Sequential(
                nn.Conv2d(a15, a32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(a32)
            )
            self.a33["stem"] = self.a34

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        a35, a36, a32 = a32, a32, self.a23

        self.a37 = nn.ModuleList()
        a38, a39 = False, False
        a40 = 0
        for a41 in range(a14):
            a38, a39 = a39, False
            if a41 in [a14 // 3, 2 * a14 // 3]:
                a32 *= 2
                a39 = True

            a42 = p(a17, a35, a36, a32, a38, a39)
            self.a37.append(a42)
            a43 = a32 * a17
            if a41 in self.a25:
                if a19 is not None:
                    self.a29[str(a41)] = a19["aux" + str(a40)]
                else:
                    self.a29[str(a41)] = self.a27(a43, self.a25[a41], self.a26)
                    self.a33["aux" + str(a40)] = self.a29[str(a41)]
                a40 += 1
            a35, a36 = a36, a43

        self.a44 = nn.AdaptiveAvgPool2d(1)
        self.a45 = nn.Linear(a36, self.a26)

    def forward(a46, a47):
        a48 = a49 = self.a34(a47)
        a50 = []

        for a51, a52 in enumerate(self.a37):
            a48, a49 = a49, a52(a48, a49)
            if str(a51) in self.a29:
                a50.append(self.a29[str(a51)](a49))

        a53 = self.a44(a49)
        a53 = a53.view(a53.size(0), -1)  # flatten
        a54 = self.a45(a53)
        a50.append(a54)

        if self.a30 is None:
            assert len(a50) == 2
            return a50[1], a50[0]
        else:
            a55 = torch.cat([(a56 * a57) for a56, a57 in zip(F.softmax(self.a30, dim=0), a50)], 0)
            return a54, a55

    def a58(a59, a60):
        for a61 in self.modules():
            if isinstance(a61, ops.DropPath):
                a61.p = a60

    def a62(a63, a64, a65):
        a66 = parse_results(a64, a63.a24)
        a65.info(a66)
        return a66