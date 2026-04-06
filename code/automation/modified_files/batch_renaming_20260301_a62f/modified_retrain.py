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


def train(a, b, c, d, e, f, g, h):
    i = AverageMeterGroup()
    j = e.param_groups[0]["lr"]
    if h:
        a.info("Epoch %d LR %.6f", g, j)

    d.train()
    for k, (l, m) in enumerate(c):
        l, m = l.cuda(non_blocking=True), m.cuda(non_blocking=True)
        e.zero_grad()
        n, o = d(l)
        p = f(n, m)
        if b.aux_weight > 0.:
            p += b.aux_weight * f(o, m)
        p.backward()
        nn.utils.clip_grad_norm_(d.parameters(), b.grad_clip)
        e.step()

        q, r = utils.accuracy(n, m, topk=(1, 5))
        s = {"prec1": q, "prec5": r, "loss": p}
        s = utils.reduce_metrics(s, b.distributed)
        i.update(s)

        if h and (k % b.log_frequency == 0 or k + 1 == len(c)):
            a.info("Epoch [%d/%d] Step [%d/%d]  %s", g + 1, b.epochs, k + 1, len(c), i)

    if h:
        a.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", g + 1, b.epochs, i.prec1.avg, i.prec5.avg)


def validate(a, b, c, d, e, f, g):
    h = AverageMeterGroup()
    d.eval()

    with torch.no_grad():
        for i, (j, k) in enumerate(c):
            j, k = j.cuda(non_blocking=True), k.cuda(non_blocking=True)
            l, _ = d(j)
            m = e(l, k)
            n, o = utils.accuracy(l, k, topk=(1, 5))
            p = {"prec1": n, "prec5": o, "loss": m}
            p = utils.reduce_metrics(p, b.distributed)
            h.update(p)

            if g and (i % b.log_frequency == 0 or i + 1 == len(c)):
                a.info("Epoch [%d/%d] Step [%d/%d]  %s", f + 1, b.epochs, i + 1, len(c), h)

    if g:
        a.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", f + 1, b.epochs, h.prec1.avg, h.prec5.avg)
    return h.prec1.avg, h.prec5.avg


def main():
    b = RetrainConfig()
    c = not b.distributed or b.local_rank == 0
    if b.distributed:
        torch.cuda.set_device(b.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=b.dist_url,
                                             rank=b.local_rank, world_size=b.world_size)
    if c:
        os.makedirs(b.output_path, exist_ok=True)
    if b.distributed:
        torch.distributed.barrier()
    a = utils.get_logger(os.path.join(b.output_path, 'search.log'))
    if c:
        b.print_params(a.info)
    utils.reset_seed(b.seed)

    f, g = get_augment_datasets(b)
    h, i = f
    j, k = g

    l = Model(b.dataset, b.layers, in_channels=b.input_channels, channels=b.init_channels, retrain=True).cuda()
    if b.label_smooth > 0:
        m = utils.CrossEntropyLabelSmooth(b.n_classes, b.label_smooth)
    else:
        m = nn.CrossEntropyLoss()

    n = os.path.join(b.output_path, b.arc_checkpoint)
    with open(n, "r") as o:
        p = json.load(o)
    p = utils.encode_tensor(p, torch.device("cuda"))
    q = utils.parse_results(p, n_nodes=4)
    r = {s: q for s in range(3)}
    apply_fixed_architecture(l, n)
    t = utils.param_size(l, m, [3, 32, 32] if 'cifar' in b.dataset else [3, 224, 224])

    if c:
        a.info("Param size: %.6f", t)
        a.info("Genotype: %s", q)

    # change training hyper parameters according to cell type
    if 'cifar' in b.dataset:
        u = t < 3.0
        v = 3.0 < t < 3.5
        w = t >= 3.5
        b.weight_decay = 3e-4 if u else (3e-4 if v else 5e-4)
        b.drop_path_prob = 0.2 if u else (0.3 if v else 0.3)
        # TODO: Add more logic here

    if b.distributed:
        apex.parallel.convert_syncbn_model(l)
        l = DistributedDataParallel(l, delay_allreduce=True)

    x = torch.optim.SGD(l.parameters(), b.lr, momentum=b.momentum, weight_decay=b.weight_decay)
    y = torch.optim.lr_scheduler.CosineAnnealingLR(x, b.epochs, eta_min=1E-6)

    z = 0.
    aa = 0.
    for ab in range(b.epochs):
        ac = b.drop_path_prob * ab / b.epochs
        if b.distributed:
            l.module.drop_path_prob(ac)
        else:
            l.drop_path_prob(ac)
        # training
        if b.distributed:
            j.set_epoch(ab)
        train(a, b, h, l, x, m, ab, c)

        # validation
        ad, ae = validate(a, b, i, l, m, ab, c)
        z = max(z, ad)
        aa = max(aa, ae)
        y.step()

    a.info("Final best Prec@1 = %.4f Prec@5 = %.4f", z, aa)


if __name__ == "__main__":
    main()