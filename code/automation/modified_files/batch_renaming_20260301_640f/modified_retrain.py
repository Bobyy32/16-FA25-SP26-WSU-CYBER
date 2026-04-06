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
    if i:
        b.info("Epoch %d LR %.6f", h, k)

    e.train()
    for l, (m, n) in enumerate(d):
        m, n = m.cuda(non_blocking=True), n.cuda(non_blocking=True)
        f.zero_grad()
        o, p = e(m)
        q = g(o, n)
        if c.aux_weight > 0.:
            q += c.aux_weight * g(p, n)
        q.backward()
        nn.utils.clip_grad_norm_(e.parameters(), c.grad_clip)
        f.step()

        r, s = utils.accuracy(o, n, topk=(1, 5))
        t = {"prec1": r, "prec5": s, "loss": q}
        t = utils.reduce_metrics(t, c.distributed)
        j.update(t)

        if i and (l % c.log_frequency == 0 or l + 1 == len(d)):
            b.info("Epoch [%d/%d] Step [%d/%d]  %s", h + 1, c.epochs, l + 1, len(d), j)

    if i:
        b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", h + 1, c.epochs, j.prec1.avg, j.prec5.avg)


def u(b, c, d, e, f, g, h, i):
    j = AverageMeterGroup()
    e.eval()

    with torch.no_grad():
        for l, (m, n) in enumerate(d):
            m, n = m.cuda(non_blocking=True), n.cuda(non_blocking=True)
            o, _ = e(m)
            q = f(o, n)
            r, s = utils.accuracy(o, n, topk=(1, 5))
            t = {"prec1": r, "prec5": s, "loss": q}
            t = utils.reduce_metrics(t, c.distributed)
            j.update(t)

            if i and (l % c.log_frequency == 0 or l + 1 == len(d)):
                b.info("Epoch [%d/%d] Step [%d/%d]  %s", g + 1, c.epochs, l + 1, len(d), j)

    if i:
        b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", g + 1, c.epochs, j.prec1.avg, j.prec5.avg)
    return j.prec1.avg, j.prec5.avg


def v():
    c = RetrainConfig()
    w = not c.distributed or c.local_rank == 0
    if c.distributed:
        torch.cuda.set_device(c.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=c.dist_url,
                                             rank=c.local_rank, world_size=c.world_size)
    if w:
        os.makedirs(c.output_path, exist_ok=True)
    if c.distributed:
        torch.distributed.barrier()
    b = utils.get_logger(os.path.join(c.output_path, 'search.log'))
    if w:
        c.print_params(b.info)
    utils.reset_seed(c.seed)

    x, y = get_augment_datasets(c)
    z, A = x
    B, C = y

    D = Model(c.dataset, c.layers, in_channels=c.input_channels, channels=c.init_channels, retrain=True).cuda()
    if c.label_smooth > 0:
        E = utils.CrossEntropyLabelSmooth(c.n_classes, c.label_smooth)
    else:
        E = nn.CrossEntropyLoss()

    F = os.path.join(c.output_path, c.arc_checkpoint)
    with open(F, "r") as G:
        H = json.load(G)
    H = utils.encode_tensor(H, torch.device("cuda"))
    I = utils.parse_results(H, n_nodes=4)
    J = {i: I for i in range(3)}
    apply_fixed_architecture(D, F)
    K = utils.param_size(D, E, [3, 32, 32] if 'cifar' in c.dataset else [3, 224, 224])

    if w:
        b.info("Param size: %.6f", K)
        b.info("Genotype: %s", I)

    # change training hyper parameters according to cell type
    if 'cifar' in c.dataset:
        L = 3.0 if K < 3.0 else (3.0 if 3.0 < K < 3.5 else 5e-4)
        M = 0.2 if K < 3.0 else (0.3 if 3.0 < K < 3.5 else 0.3)
        c.weight_decay = L
        c.drop_path_prob = M

    if c.distributed:
        apex.parallel.convert_syncbn_model(D)
        D = DistributedDataParallel(D, delay_allreduce=True)

    N = torch.optim.SGD(D.parameters(), c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    O = torch.optim.lr_scheduler.CosineAnnealingLR(N, c.epochs, eta_min=1E-6)

    P = Q = 0.
    for R in range(c.epochs):
        S = c.drop_path_prob * R / c.epochs
        if c.distributed:
            D.module.drop_path_prob(S)
        else:
            D.drop_path_prob(S)
        # training
        if c.distributed:
            B.set_epoch(R)
        a(b, c, z, D, N, E, R, w)

        # validation
        T, U = u(b, c, A, D, E, R, w)
        P = T if T > P else P
        Q = U if U > Q else Q
        O.step()

    b.info("Final best Prec@1 = %.4f Prec@5 = %.4f", P, Q)


if __name__ == "__main__":
    v()