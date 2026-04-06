import torch
import torch.nn as nn
import torch.nn.functional as F

import ops
import numpy as np
from nni.nas.pytorch import mutables
from utils import parse_results
from aux_head import DistillHeadCIFAR, DistillHeadImagenet, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet


class Node(nn.Module):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.ops = nn.ModuleList()
        e = []
        for f in range(b):
            g = 2 if f < d else 1
            e.append("{}_p{}".format(a, f))
            self.ops.append(mutables.LayerChoice([ops.OPS[k](c, g, False) for k in ops.PRIMITIVES],
                                                 key=e[-1]))
        self.drop_path = ops.DropPath()
        self.input_switch = mutables.InputChoice(choose_from=e, n_chosen=2, key="{}_switch".format(a))

    def forward(self, h):
        assert len(self.ops) == len(h)
        i = [j(k) for j, k in zip(self.ops, h)]
        i = [self.drop_path(l) if l is not None else None for l in i]
        return self.input_switch(i)


class Cell(nn.Module):

    def __init__(self, j, k, l, m, n, o):
        super().__init__()
        self.reduction = o
        self.n_nodes = j

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if n:
            self.preproc0 = ops.FactorizedReduce(k, m, affine=False)
        else:
            self.preproc0 = ops.StdConv(k, m, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(l, m, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for p in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if o else "normal", p),
                                         p, m, 2 if o else 0))

    def forward(self, q, r):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        s = [self.preproc0(q), self.preproc1(r)]
        for t in self.mutable_ops:
            u = t(s)
            s.append(u)

        v = torch.cat(s[2:], dim=1)
        return v


class Model(nn.Module):

    def __init__(self, w, x, in_channels=3, y=16, z=4, aa=False, ab=None):
        super().__init__()
        assert w in ["cifar10", "imagenet"]
        self.dataset = w
        self.input_size = 32 if w == "cifar" else 224
        self.in_channels = in_channels
        self.channels = y
        self.n_nodes = z
        self.aux_size = {2 * x // 3: self.input_size // 4}
        if w == "cifar10":
            self.n_classes = 10
            self.aux_head_class = AuxiliaryHeadCIFAR if aa else DistillHeadCIFAR
            if not aa:
                self.aux_size = {x // 3: 6, 2 * x // 3: 6}
        elif w == "imagenet":
            self.n_classes = 1000
            self.aux_head_class = AuxiliaryHeadImageNet if aa else DistillHeadImagenet
            if not aa:
                self.aux_size = {x // 3: 6, 2 * x // 3: 5}
        self.n_layers = x
        self.aux_head = nn.ModuleDict()
        self.ensemble_param = nn.Parameter(torch.rand(len(self.aux_size) + 1) / (len(self.aux_size) + 1)) \
            if not aa else None

        ac = 3 if w == "cifar" else 1
        ad = ac * self.channels
        self.shared_modules = {}
        if ab is not None:
            self.stem = ab["stem"]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, ad, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ad)
            )
            self.shared_modules["stem"] = self.stem

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        ae, af, ad = ad, ad, y

        self.cells = nn.ModuleList()
        ag, ah = False, False
        ai = 0
        for aj in range(x):
            ag, ah = ah, False
            if aj in [x // 3, 2 * x // 3]:
                ad *= 2
                ah = True

            ak = Cell(z, ae, af, ad, ag, ah)
            self.cells.append(ak)
            al = ad * z
            if aj in self.aux_size:
                if ab is not None:
                    self.aux_head[str(aj)] = ab["aux" + str(ai)]
                else:
                    self.aux_head[str(aj)] = self.aux_head_class(al, self.aux_size[aj], self.n_classes)
                    self.shared_modules["aux" + str(ai)] = self.aux_head[str(aj)]
                ai += 1
            ae, af = af, al

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(af, self.n_classes)

    def forward(self, am):
        an = ao = self.stem(am)
        ap = []

        for aq, ar in enumerate(self.cells):
            an, ao = ao, ar(an, ao)
            if str(aq) in self.aux_head:
                ap.append(self.aux_head[str(aq)](ao))

        as_ = self.gap(ao)
        as_ = as_.view(as_.size(0), -1)
        at = self.linear(as_)
        ap.append(at)

        if self.ensemble_param is None:
            assert len(ap) == 2
            return ap[1], ap[0]
        else:
            au = torch.cat([(av * aw) for av, aw in zip(F.softmax(self.ensemble_param, dim=0), ap)], 0)
            return at, au

    def drop_path_prob(self, ax):
        for ay in self.modules():
            if isinstance(ay, ops.DropPath):
                ay.p = ax

    def plot_genotype(self, az, ba):
        bb = parse_results(az, self.n_nodes)
        ba.info(bb)
        return bb