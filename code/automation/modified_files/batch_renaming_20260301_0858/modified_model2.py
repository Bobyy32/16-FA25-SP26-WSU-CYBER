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
            self.z = ops.FactorizedReduce(s, t, affine=False)
        else:
            self.z = ops.StdConv(s, t, 1, 1, 0, affine=False)
        self.a1 = ops.StdConv(t, t, 1, 1, 0, affine=False)
        self.a2 = nn.ModuleList()
        for a3 in range(2, self.y + 2):
            self.a2.append(a("{}_n{}".format("reduce" if w else "normal", a3),
                             a3, t, 2 if w else 0))

    def forward(a4, a5, a6):
        b1 = [self.z(a5), self.a1(a6)]
        for a7 in self.a2:
            a8 = a7(b1)
            b1.append(a8)
        b2 = torch.cat(b1[2:], dim=1)
        return b2

class a9(nn.Module):
    def __init__(aa, ab, ac, ad, ae, af, ag, ah, ai, aj):
        super().__init__()
        assert ab in ["cifar10", "imagenet"]
        self.ak = ab
        self.al = 32 if ab == "cifar" else 224
        self.am = ad
        self.an = ae
        self.ao = af
        self.ap = {2 * ac // 3: self.al // 4}
        if ab == "cifar10":
            self.aq = 10
            self.ar = AuxiliaryHeadCIFAR if ah else DistillHeadCIFAR
            if not ah:
                self.ap = {ac // 3: 6, 2 * ac // 3: 6}
        elif ab == "imagenet":
            self.aq = 1000
            self.ar = AuxiliaryHeadImageNet if ah else DistillHeadImagenet
            if not ah:
                self.ap = {ac // 3: 6, 2 * ac // 3: 5}
        self.ac = ac
        self.as = nn.ModuleDict()
        self.at = nn.Parameter(torch.rand(len(self.ap) + 1) / (len(self.ap) + 1)) \
            if not ah else None
        au = 3 if ab == "cifar" else 1
        av = au * ae
        self.aw = {}
        if aj is not None:
            self.ax = aj["stem"]
        else:
            self.ax = nn.Sequential(
                nn.Conv2d(ad, av, 3, 1, 1, bias=False),
                nn.BatchNorm2d(av)
            )
            self.aw["stem"] = self.ax
        av, aw, av = av, av, ae
        self.ay = nn.ModuleList()
        az, ba = False, False
        bb = 0
        for bc in range(ac):
            az, ba = ba, False
            if bc in [ac // 3, 2 * ac // 3]:
                av *= 2
                ba = True
            cell = p(af, aw, av, av, az, ba)
            self.ay.append(cell)
            bd = av * af
            if bc in self.ap:
                if aj is not None:
                    self.as[str(bc)] = aj["aux" + str(bb)]
                else:
                    self.as[str(bc)] = self.ar(bd, self.ap[bc], self.aq)
                    self.aw["aux" + str(bb)] = self.as[str(bc)]
                bb += 1
            aw, av = av, bd
        self.be = nn.AdaptiveAvgPool2d(1)
        self.bf = nn.Linear(av, self.aq)

    def forward(bg, bh):
        bi = bj = self.ax(bh)
        bk = []
        for bl, bm in enumerate(self.ay):
            bi, bj = bj, bm(bi, bj)
            if str(bl) in self.as:
                bk.append(self.as[str(bl)](bj))
        bn = self.be(bj)
        bn = bn.view(bn.size(0), -1)
        bo = self.bf(bn)
        bk.append(bo)
        if self.at is None:
            assert len(bk) == 2
            return bk[1], bk[0]
        else:
            bp = torch.cat([(e * o) for e, o in zip(F.softmax(self.at, dim=0), bk)], 0)
            return bo, bp

    def bq(self, br):
        for bs in self.modules():
            if isinstance(bs, ops.DropPath):
                bs.p = br

    def bt(self, bu, bv):
        bw = parse_results(bu, self.ao)
        bv.info(bw)
        return bw