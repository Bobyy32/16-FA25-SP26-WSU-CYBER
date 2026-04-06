import json
import logging
import os
import time
from argparse import ArgumentParser
import torch
import torch.nn as nn
import apex
import datasets
import utils
from apex.parallel import DistributedDataParallel
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
        r = g(p, o)
        r += c.aux_weight * g(q, o) if c.aux_weight > 0. else r
        r.backward()
        nn.utils.clip_grad_norm_(e.parameters(), c.grad_clip)
        f.step()
        s, t = utils.accuracy(p, o, topk=(1, 5))
        u = {"prec1": s, "prec5": t, "loss": r}
        u = utils.reduce_metrics(u, c.distributed)
        j.update(u)
        v = h and (m % c.log_frequency == 0 or m + 1 == len(d))
        w = b.info("Epoch [%d/%d] Step [%d/%d]  %s", i + 1, c.epochs, m + 1, len(d), j) if v else None
    x = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", i + 1, c.epochs, j.prec1.avg, j.prec5.avg) if h else None

def b(c, d, e, f, g, h, i):
    j = AverageMeterGroup()
    f.eval()
    with torch.no_grad():
        for k, (l, m) in enumerate(e):
            l, m = l.cuda(non_blocking=True), m.cuda(non_blocking=True)
            n, _ = f(l)
            o = g(n, m)
            p, q = utils.accuracy(n, m, topk=(1, 5))
            r = {"prec1": p, "prec5": q, "loss": o}
            r = utils.reduce_metrics(r, d.distributed)
            j.update(r)
            s = i and (k % d.log_frequency == 0 or k + 1 == len(e))
            t = c.info("Epoch [%d/%d] Step [%d/%d]  %s", h + 1, d.epochs, k + 1, len(e), j) if s else None
    u = c.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", h + 1, d.epochs, j.prec1.avg, j.prec5.avg) if i else None
    return j.prec1.avg, j.prec5.avg

def c():
    d = RetrainConfig()
    e = not d.distributed or d.local_rank == 0
    if d.distributed:
        torch.cuda.set_device(d.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=d.dist_url, rank=d.local_rank, world_size=d.world_size)
    f = os.makedirs(d.output_path, exist_ok=True) if e else None
    if d.distributed:
        torch.distributed.barrier()
    g = utils.get_logger(os.path.join(d.output_path, 'search.log'))
    if e:
        d.print_params(g.info)
    utils.reset_seed(d.seed)
    h, i = get_augment_datasets(d)
    j, k = h
    l, m = i
    n = Model(d.dataset, d.layers, in_channels=d.input_channels, channels=d.init_channels, retrain=True).cuda()
    o = utils.CrossEntropyLabelSmooth(d.n_classes, d.label_smooth) if d.label_smooth > 0 else nn.CrossEntropyLoss()
    p = os.path.join(d.output_path, d.arc_checkpoint)
    with open(p, "r") as q:
        r = json.load(q)
    r = utils.encode_tensor(r, torch.device("cuda"))
    s = utils.parse_results(r, n_nodes=4)
    t = {u: s for u in range(3)}
    apply_fixed_architecture(n, p)
    v = utils.param_size(n, o, [3, 32, 32] if 'cifar' in d.dataset else [3, 224, 224])
    if e:
        g.info("Param size: %.6f", v)
        g.info("Genotype: %s", s)
    w = 3.0 if 'cifar' in d.dataset else (3.0 < v < 3.5 if d.weight_decay = 3e-4 else (v >= 3.5 if d.weight_decay = 5e-4 else None))
    if d.distributed:
        apex.parallel.convert_syncbn_model(n)
        n = DistributedDataParallel(n, delay_allreduce=True)
    x = torch.optim.SGD(n.parameters(), d.lr, momentum=d.momentum, weight_decay=d.weight_decay)
    y = torch.optim.lr_scheduler.CosineAnnealingLR(x, d.epochs, eta_min=1E-6)
    z = 0.
    A = 0.
    for B in range(d.epochs):
        C = d.drop_path_prob * B / d.epochs
        if d.distributed:
            n.module.drop_path_prob(C)
        else:
            n.drop_path_prob(C)
        if d.distributed:
            l.set_epoch(B)
        a(g, d, j, n, x, o, B, e)
        D, E = b(g, d, k, n, o, B, e)
        z = max(z, D)
        A = max(A, E)
        y.step()
    g.info("Final best Prec@1 = %.4f Prec@5 = %.4f", z, A)