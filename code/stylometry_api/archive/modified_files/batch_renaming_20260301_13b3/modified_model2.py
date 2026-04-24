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
        self.A = ops.StdConv(t, t, 1, 1, 0, affine=False)
        self.B = nn.ModuleList()
        for C in range(2, self.y + 2):
            self.B.append(a("{}_n{}".format("reduce" if w else "normal", C),
                                         C, t, 2 if w else 0))

    def forward(D, E, F):
        G = [self.z(E), self.A(F)]
        for H in self.B:
            I = H(G)
            G.append(I)
        J = torch.cat(G[2:], dim=1)
        return J

class K(nn.Module):
    def __init__(L, M, N, O, P, Q, R, S, T, U, V, W):
        super().__init__()
        assert M in ["cifar10", "imagenet"]
        self.X = M
        self.Y = 32 if M == "cifar" else 224
        self.Z = O
        self.a1 = P
        self.a2 = R
        self.a3 = {2 * N // 3: self.Y // 4}
        if M == "cifar10":
            self.a4 = 10
            self.a5 = AuxiliaryHeadCIFAR if S else DistillHeadCIFAR
            if not S:
                self.a3 = {N // 3: 6, 2 * N // 3: 6}
        elif M == "imagenet":
            self.a4 = 1000
            self.a5 = AuxiliaryHeadImageNet if S else DistillHeadImagenet
            if not S:
                self.a3 = {N // 3: 6, 2 * N // 3: 5}
        self.a6 = N
        self.a7 = nn.ModuleDict()
        self.a8 = nn.Parameter(torch.rand(len(self.a3) + 1) / (len(self.a3) + 1)) \
            if not S else None
        stem_multiplier = 3 if M == "cifar" else 1
        c_cur = stem_multiplier * P
        self.a9 = {}
        if V is not None:
            self.aa = V["stem"]
        else:
            self.aa = nn.Sequential(
                nn.Conv2d(O, c_cur, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_cur)
            )
            self.a9["stem"] = self.aa
        channels_pp, channels_p, c_cur = c_cur, c_cur, P
        self.ab = nn.ModuleList()
        reduction_p, reduction = False, False
        aux_head_count = 0
        for i in range(N):
            reduction_p, reduction = reduction, False
            if i in [N // 3, 2 * N // 3]:
                c_cur *= 2
                reduction = True
            cell = p(R, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.ab.append(cell)
            c_cur_out = c_cur * R
            if i in self.a3:
                if V is not None:
                    self.a7[str(i)] = V["aux" + str(aux_head_count)]
                else:
                    self.a7[str(i)] = self.a5(c_cur_out, self.a3[i], self.a4)
                    self.a9["aux" + str(aux_head_count)] = self.a7[str(i)]
                aux_head_count += 1
            channels_pp, channels_p = channels_p, c_cur_out
        self.ac = nn.AdaptiveAvgPool2d(1)
        self.ad = nn.Linear(channels_p, self.a4)

    def forward(ae, af):
        ag = ah = self.aa(af)
        ai = []
        for i, aj in enumerate(self.ab):
            ag, ah = ah, aj(ag, ah)
            if str(i) in self.a7:
                ai.append(self.a7[str(i)](ah))
        ak = self.ac(ah)
        ak = ak.view(ak.size(0), -1)
        al = self.ad(ak)
        ai.append(al)
        if self.a8 is None:
            assert len(ai) == 2
            return ai[1], ai[0]
        else:
            am = torch.cat([(e * o) for e, o in zip(F.softmax(self.a8, dim=0), ai)], 0)
            return al, am

    def an(self, ao):
        for ap in self.modules():
            if isinstance(ap, ops.DropPath):
                ap.p = ao

    def aq(self, ar, as):
        at = parse_results(ar, self.a2)
        as.info(at)
        return at