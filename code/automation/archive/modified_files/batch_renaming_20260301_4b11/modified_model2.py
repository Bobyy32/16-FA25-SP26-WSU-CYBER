import torch
import torch.nn as nn
import torch.nn.functional as F

import ops
import numpy as np
from nni.nas.pytorch import mutables
from utils import parse_results
from aux_head import DistillHeadCIFAR, DistillHeadImagenet, AuxiliaryHeadCIFAR, AuxiliaryHeadImageNet


class Node(nn.Module):
    def __init__(a, b, c, d):
        super().__init__()
        a.ops = nn.ModuleList()
        e = []
        for f in range(c):
            g = 2 if f < d else 1
            e.append("{}_p{}".format(b, f))
            a.ops.append(mutables.LayerChoice([ops.OPS[k](a.channels, g, False) for k in ops.PRIMITIVES],
                                                 key=e[-1]))
        a.drop_path = ops.DropPath()
        a.input_switch = mutables.InputChoice(choose_from=e, n_chosen=2, key="{}_switch".format(b))

    def forward(a, b):
        assert len(a.ops) == len(b)
        c = [op(d) for op, d in zip(a.ops, b)]
        c = [a.drop_path(o) if o is not None else None for o in c]
        return a.input_switch(c)


class Cell(nn.Module):

    def __init__(a, b, c, d, e, f):
        super().__init__()
        a.reduction = f
        a.n_nodes = b

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        a.preproc0 = ops.FactorizedReduce(c, d, affine=False) if e else ops.StdConv(c, d, 1, 1, 0, affine=False)
        a.preproc1 = ops.StdConv(d, d, 1, 1, 0, affine=False)

        # generate dag
        a.mutable_ops = nn.ModuleList()
        for g in range(2, a.n_nodes + 2):
            a.mutable_ops.append(Node("{}_n{}".format("reduce" if f else "normal", g),
                                         g, d, 2 if f else 0))

    def forward(a, b, c):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        d = [a.preproc0(b), a.preproc1(c)]
        for e in a.mutable_ops:
            f = e(d)
            d.append(f)

        g = torch.cat(d[2:], dim=1)
        return g


class Model(nn.Module):

    def __init__(a, b, c, d=3, e=16, f=4, g=False, h=None):
        super().__init__()
        assert b in ["cifar10", "imagenet"]
        a.dataset = b
        a.input_size = 32 if b == "cifar" else 224
        a.in_channels = d
        a.channels = e
        a.n_nodes = f
        a.aux_size = {2 * c // 3: a.input_size // 4}
        a.n_classes = 10 if b == "cifar10" else 1000
        a.aux_head_class = AuxiliaryHeadCIFAR if g else DistillHeadCIFAR if b == "cifar10" else AuxiliaryHeadImageNet if g else DistillHeadImagenet
        a.n_layers = c
        a.aux_head = nn.ModuleDict()
        a.ensemble_param = nn.Parameter(torch.rand(len(a.aux_size) + 1) / (len(a.aux_size) + 1)) if not g else None

        i = 3 if b == "cifar" else 1
        j = i * e
        a.shared_modules = {}  # do not wrap with ModuleDict
        if h is not None:
            a.stem = h["stem"]
        else:
            a.stem = nn.Sequential(
                nn.Conv2d(d, j, 3, 1, 1, bias=False),
                nn.BatchNorm2d(j)
            )
            a.shared_modules["stem"] = a.stem

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        k, l, j = j, j, e

        a.cells = nn.ModuleList()
        m, n = False, False
        o = 0
        for p in range(c):
            m, n = n, False
            if p in [c // 3, 2 * c // 3]:
                j *= 2
                n = True

            q = Cell(f, k, l, j, m, n)
            a.cells.append(q)
            r = j * f
            if p in a.aux_size:
                if h is not None:
                    a.aux_head[str(p)] = h["aux" + str(o)]
                else:
                    a.aux_head[str(p)] = a.aux_head_class(r, a.aux_size[p], a.n_classes)
                    a.shared_modules["aux" + str(o)] = a.aux_head[str(p)]
                o += 1
            k, l = l, r

        a.gap = nn.AdaptiveAvgPool2d(1)
        a.linear = nn.Linear(l, a.n_classes)

    def forward(a, b):
        c = d = a.stem(b)
        e = []

        for f, g in enumerate(a.cells):
            c, d = d, g(c, d)
            if str(f) in a.aux_head:
                e.append(a.aux_head[str(f)](d))

        h = a.gap(d)
        h = h.view(h.size(0), -1)  # flatten
        i = a.linear(h)
        e.append(i)

        if a.ensemble_param is None:
            assert len(e) == 2
            return e[1], e[0]
        else:
            j = torch.cat([(k * l) for k, l in zip(F.softmax(a.ensemble_param, dim=0), e)], 0)
            return i, j

    def drop_path_prob(a, b):
        for c in a.modules():
            if isinstance(c, ops.DropPath):
                c.p = b

    def plot_genotype(a, b, c):
        d = parse_results(b, a.n_nodes)
        c.info(d)
        return d