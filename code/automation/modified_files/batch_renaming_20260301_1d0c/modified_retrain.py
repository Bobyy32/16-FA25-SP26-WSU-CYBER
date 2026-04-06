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
        r = r + c.aux_weight * g(q, o) if c.aux_weight > 0. else r
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

def y(z, c, d, e, f, g, h):
    j = AverageMeterGroup()
    e.eval()
    with torch.no_grad():
        for m, (n, o) in enumerate(d):
            n, o = n.cuda(non_blocking=True), o.cuda(non_blocking=True)
            p, _ = e(n)
            r = f(p, o)
            s, t = utils.accuracy(p, o, topk=(1, 5))
            u = {"prec1": s, "prec5": t, "loss": r}
            u = utils.reduce_metrics(u, c.distributed)
            j.update(u)
            v = h and (m % c.log_frequency == 0 or m + 1 == len(d))
            w = z.info("Epoch [%d/%d] Step [%d/%d]  %s", g + 1, c.epochs, m + 1, len(d), j) if v else None
    x = z.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", g + 1, c.epochs, j.prec1.avg, j.prec5.avg) if h else None
    return j.prec1.avg, j.prec5.avg

def p():
    c = RetrainConfig()
    h = not c.distributed or c.local_rank == 0
    if c.distributed:
        torch.cuda.set_device(c.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=c.dist_url, rank=c.local_rank, world_size=c.world_size)
    if h:
        os.makedirs(c.output_path, exist_ok=True)
    if c.distributed:
        torch.distributed.barrier()
    b = utils.get_logger(os.path.join(c.output_path, 'search.log'))
    if h:
        c.print_params(b.info)
    utils.reset_seed(c.seed)
    f, g = get_augment_datasets(c)
    i, k = f
    l, m = g
    e = Model(c.dataset, c.layers, in_channels=c.input_channels, channels=c.init_channels, retrain=True).cuda()
    if c.label_smooth > 0:
        g = utils.CrossEntropyLabelSmooth(c.n_classes, c.label_smooth)
    else:
        g = nn.CrossEntropyLoss()
    o = os.path.join(c.output_path, c.arc_checkpoint)
    with open(o, "r") as q:
        r = json.load(q)
    r = utils.encode_tensor(r, torch.device("cuda"))
    s = utils.parse_results(r, n_nodes=4)
    t = {u: s for u in range(3)}
    apply_fixed_architecture(e, o)
    v = utils.param_size(e, g, [3, 32, 32] if 'cifar' in c.dataset else [3, 224, 224])
    if h:
        b.info("Param size: %.6f", v)
        b.info("Genotype: %s", s)
    w = 'cifar' in c.dataset
    x = 3.0 < v < 3.5
    y = v >= 3.5
    z = c.weight_decay = 3e-4 if w and v < 3.0 else c.weight_decay = 3e-4 if w and x else c.weight_decay = 5e-4
    c.drop_path_prob = 0.2 if w and v < 3.0 else c.drop_path_prob = 0.3 if w and x else c.drop_path_prob = 0.3
    if c.distributed:
        apex.parallel.convert_syncbn_model(e)
        e = DistributedDataParallel(e, delay_allreduce=True)
    f = torch.optim.SGD(e.parameters(), c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    u = torch.optim.lr_scheduler.CosineAnnealingLR(f, c.epochs, eta_min=1E-6)
    a = b = 0.
    for i in range(c.epochs):
        d = c.drop_path_prob * i / c.epochs
        if c.distributed:
            e.module.drop_path_prob(d)
        else:
            e.drop_path_prob(d)
        if c.distributed:
            l.set_epoch(i)
        a(b, c, i, e, f, g, i, h)
        j, k = y(b, c, k, e, g, i, h)
        a = max(a, j)
        b = max(b, k)
        u.step()
    b.info("Final best Prec@1 = %.4f Prec@5 = %.4f", a, b)

if __name__ == "__main__":
    p()