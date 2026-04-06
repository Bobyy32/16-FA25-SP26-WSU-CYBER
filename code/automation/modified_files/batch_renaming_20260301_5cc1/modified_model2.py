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

    def __init__(self, m, n, o, p, q, r):
        super().__init__()
        self.reduction = r
        self.n_nodes = m

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if q:
            self.preproc0 = ops.FactorizedReduce(n, p, affine=False)
        else:
            self.preproc0 = ops.StdConv(n, p, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(o, p, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for s in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if r else "normal", s),
                                         s, p, 2 if r else 0))

    def forward(self, t, u):
        # t, u are the outputs of previous previous cell and previous cell, respectively.
        v = [self.preproc0(t), self.preproc1(u)]
        for w in self.mutable_ops:
            x = w(v)
            v.append(x)

        y = torch.cat(v[2:], dim=1)
        return y


class Model(nn.Module):

    def __init__(self, z, a1, in_channels=3, channels=16, n_nodes=4, retrain=False, shared_modules=None):
        super().__init__()
        assert z in ["cifar10", "imagenet"]
        self.dataset = z
        self.input_size = 32 if z == "cifar" else 224
        self.in_channels = in_channels
        self.channels = channels
        self.n_nodes = n_nodes
        self.aux_size = {2 * a1 // 3: self.input_size // 4}
        if z == "cifar10":
            self.n_classes = 10
            self.aux_head_class = AuxiliaryHeadCIFAR if retrain else DistillHeadCIFAR
            if not retrain:
                self.aux_size = {a1 // 3: 6, 2 * a1 // 3: 6}
        elif z == "imagenet":
            self.n_classes = 1000
            self.aux_head_class = AuxiliaryHeadImageNet if retrain else DistillHeadImagenet
            if not retrain:
                self.aux_size = {a1 // 3: 6, 2 * a1 // 3: 5}
        self.n_layers = a1
        self.aux_head = nn.ModuleDict()
        self.ensemble_param = nn.Parameter(torch.rand(len(self.aux_size) + 1) / (len(self.aux_size) + 1)) \
            if not retrain else None

        stem_multiplier = 3 if z == "cifar" else 1
        c_cur = stem_multiplier * self.channels
        self.shared_modules = {}  # do not wrap with ModuleDict
        if shared_modules is not None:
            self.stem = shared_modules["stem"]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c_cur)
            )
            self.shared_modules["stem"] = self.stem

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        aux_head_count = 0
        for i in range(a1):
            reduction_p, reduction = reduction, False
            if i in [a1 // 3, 2 * a1 // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            if i in self.aux_size:
                if shared_modules is not None:
                    self.aux_head[str(i)] = shared_modules["aux" + str(aux_head_count)]
                else:
                    self.aux_head[str(i)] = self.aux_head_class(c_cur_out, self.aux_size[i], self.n_classes)
                    self.shared_modules["aux" + str(aux_head_count)] = self.aux_head[str(i)]
                aux_head_count += 1
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, self.n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        outputs = []

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if str(i) in self.aux_head:
                outputs.append(self.aux_head[str(i)](s1))

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        outputs.append(logits)

        if self.ensemble_param is None:
            assert len(outputs) == 2
            return outputs[1], outputs[0]
        else:
            em_output = torch.cat([(e * o) for e, o in zip(F.softmax(self.ensemble_param, dim=0), outputs)], 0)
            return logits, em_output

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p

    def plot_genotype(self, results, logger):
        genotypes = parse_results(results, self.n_nodes)
        logger.info(genotypes)
        return genotypes