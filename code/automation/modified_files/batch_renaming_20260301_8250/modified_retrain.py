# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

import apex  # pylint: disable=import-error
import datasets
import utils
from apex.parallel import DistributedDataParallel  # pylint: disable=import-error
from config import RetrainConfig
from datasets.cifar import get_augment_datasets
from model import Model
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeterGroup


def a(b, c, d, e, f, g, h, i):
    j = AverageMeterGroup()
    k = f.param_groups[0]["lr"]
    l = b.info("Epoch %d LR %.6f", i, k) if h else None
    e.train()
    for m, (n, o) in enumerate(d):
        n, o = n.cuda(non_blocking=True), o.cuda(non_blocking=True)
        f.zero_grad()
        p, q = e(n)
        r = c(p, o)
        r += c(q, o) * g.aux_weight if g.aux_weight > 0. else r
        r.backward()
        nn.utils.clip_grad_norm_(e.parameters(), g.grad_clip)
        f.step()
        s, t = utils.accuracy(p, o, topk=(1, 5))
        u = {"prec1": s, "prec5": t, "loss": r}
        u = utils.reduce_metrics(u, g.distributed)
        j.update(u)
        v = h and (m % g.log_frequency == 0 or m + 1 == len(d))
        w = b.info("Epoch [%d/%d] Step [%d/%d]  %s", i + 1, g.epochs, m + 1, len(d), j) if v else None
    x = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", i + 1, g.epochs, j.prec1.avg, j.prec5.avg) if h else None


def y(z, aa, ab, ac, ad, ae, af):
    ag = AverageMeterGroup()
    ac.eval()
    with torch.no_grad():
        for ah, (ai, aj) in enumerate(ab):
            ai, aj = ai.cuda(non_blocking=True), aj.cuda(non_blocking=True)
            ak, _ = ac(ai)
            al = ad(ak, aj)
            am, an = utils.accuracy(ak, aj, topk=(1, 5))
            ao = {"prec1": am, "prec5": an, "loss": al}
            ao = utils.reduce_metrics(ao, af.distributed)
            ag.update(ao)
            ap = af.main_proc and (ah % af.log_frequency == 0 or ah + 1 == len(ab))
            aq = z.info("Epoch [%d/%d] Step [%d/%d]  %s", ae + 1, af.epochs, ah + 1, len(ab), ag) if ap else None
    ar = z.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", ae + 1, af.epochs, ag.prec1.avg, ag.prec5.avg) if af.main_proc else None
    return ag.prec1.avg, ag.prec5.avg


def asd():
    af = RetrainConfig()
    at = not af.distributed or af.local_rank == 0
    if af.distributed:
        torch.cuda.set_device(af.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=af.dist_url, rank=af.local_rank, world_size=af.world_size)
    au = at and os.makedirs(af.output_path, exist_ok=True)
    if af.distributed:
        torch.distributed.barrier()
    av = utils.get_logger(os.path.join(af.output_path, 'search.log'))
    aw = at and af.print_params(av.info)
    utils.reset_seed(af.seed)
    ax, ay = get_augment_datasets(af)
    az, ba = ax
    bb, bc = ay
    bd = Model(af.dataset, af.layers, in_channels=af.input_channels, channels=af.init_channels, retrain=True).cuda()
    be = utils.CrossEntropyLabelSmooth(af.n_classes, af.label_smooth) if af.label_smooth > 0 else nn.CrossEntropyLoss()
    bf = os.path.join(af.output_path, af.arc_checkpoint)
    with open(bf, "r") as bg:
        bh = json.load(bg)
    bh = utils.encode_tensor(bh, torch.device("cuda"))
    bi = utils.parse_results(bh, n_nodes=4)
    bj = {i: bi for i in range(3)}
    apply_fixed_architecture(bd, bf)
    bk = utils.param_size(bd, be, [3, 32, 32] if 'cifar' in af.dataset else [3, 224, 224])
    bl = at and av.info("Param size: %.6f", bk)
    bm = at and av.info("Genotype: %s", bi)
    bn = 'cifar' in af.dataset
    bo = 3.0 < bk < 3.5
    bp = 3.0 > bk
    bq = 3.5 < bk
    br = 3e-4 if bp else (5e-4 if bq else 3e-4)
    bs = 0.2 if bp else (0.3 if bq else 0.3)
    af.weight_decay = br
    af.drop_path_prob = bs
    if af.distributed:
        apex.parallel.convert_syncbn_model(bd)
        bd = DistributedDataParallel(bd, delay_allreduce=True)
    bt = torch.optim.SGD(bd.parameters(), af.lr, momentum=af.momentum, weight_decay=af.weight_decay)
    bu = torch.optim.lr_scheduler.CosineAnnealingLR(bt, af.epochs, eta_min=1E-6)
    bv = 0.
    bw = 0.
    for bx in range(af.epochs):
        by = af.drop_path_prob * bx / af.epochs
        bz = bd.module.drop_path_prob(by) if af.distributed else bd.drop_path_prob(by)
        bA = af.distributed
        bB = bA and bb.set_epoch(bx)
        a(av, af, az, bd, bt, be, bx, at)
        bC, bD = y(av, af, bc, bd, be, bx, at)
        bv = bC if bC > bv else bv
        bw = bD if bD > bw else bw
        bu.step()
    av.info("Final best Prec@1 = %.4f Prec@5 = %.4f", bv, bw)


if __name__ == "__main__":
    asd()