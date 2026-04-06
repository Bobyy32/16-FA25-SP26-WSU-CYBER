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

def a(b, c, d, e, f, g, h, i, j):
    k = AverageMeterGroup()
    l = f.param_groups[0]["lr"]
    m = "Epoch %d LR %.6f" % (i, l) if j else None
    e.train()
    for n, (o, p) in enumerate(d):
        o, p = o.cuda(non_blocking=True), p.cuda(non_blocking=True)
        f.zero_grad()
        q, r = e(o)
        s = c(q, p)
        s += c(r, p) if g.aux_weight > 0. else s
        s.backward()
        nn.utils.clip_grad_norm_(e.parameters(), g.grad_clip)
        f.step()
        t, u = utils.accuracy(q, p, topk=(1, 5))
        v = {"prec1": t, "prec5": u, "loss": s}
        v = utils.reduce_metrics(v, g.distributed)
        k.update(v)
        w = "Epoch [%d/%d] Step [%d/%d]  %s" % (i + 1, g.epochs, n + 1, len(d), k) if j and (n % g.log_frequency == 0 or n + 1 == len(d)) else None
    x = "Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f" % (i + 1, g.epochs, k.prec1.avg, k.prec5.avg) if j else None

def y(b, c, d, e, f, g, h, i):
    k = AverageMeterGroup()
    e.eval()
    with torch.no_grad():
        for n, (o, p) in enumerate(d):
            o, p = o.cuda(non_blocking=True), p.cuda(non_blocking=True)
            q, _ = e(o)
            r = c(q, p)
            s, t = utils.accuracy(q, p, topk=(1, 5))
            u = {"prec1": s, "prec5": t, "loss": r}
            u = utils.reduce_metrics(u, g.distributed)
            k.update(u)
            v = "Epoch [%d/%d] Step [%d/%d]  %s" % (h + 1, g.epochs, n + 1, len(d), k) if i and (n % g.log_frequency == 0 or n + 1 == len(d)) else None
    w = "Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f" % (h + 1, g.epochs, k.prec1.avg, k.prec5.avg) if i else None
    return k.prec1.avg, k.prec5.avg

def z():
    a = RetrainConfig()
    b = not a.distributed or a.local_rank == 0
    if a.distributed:
        torch.cuda.set_device(a.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=a.dist_url, rank=a.local_rank, world_size=a.world_size)
    if b:
        os.makedirs(a.output_path, exist_ok=True)
    if a.distributed:
        torch.distributed.barrier()
    c = utils.get_logger(os.path.join(a.output_path, 'search.log'))
    if b:
        a.print_params(c.info)
    utils.reset_seed(a.seed)
    d, e = get_augment_datasets(a)
    f, g = d
    h, i = e
    j = Model(a.dataset, a.layers, in_channels=a.input_channels, channels=a.init_channels, retrain=True).cuda()
    k = utils.CrossEntropyLabelSmooth(a.n_classes, a.label_smooth) if a.label_smooth > 0 else nn.CrossEntropyLoss()
    l = os.path.join(a.output_path, a.arc_checkpoint)
    with open(l, "r") as m:
        n = json.load(m)
    n = utils.encode_tensor(n, torch.device("cuda"))
    o = utils.parse_results(n, n_nodes=4)
    p = {q: o for q in range(3)}
    apply_fixed_architecture(j, l)
    r = utils.param_size(j, k, [3, 32, 32] if 'cifar' in a.dataset else [3, 224, 224])
    if b:
        c.info("Param size: %.6f", r)
        c.info("Genotype: %s", o)
    s = 3.0 if 'cifar' in a.dataset else 3.0
    t = 3.5 if 'cifar' in a.dataset else 3.5
    a.weight_decay = 3e-4 if r < s else 5e-4 if r >= t else 3e-4
    a.drop_path_prob = 0.2 if r < s else 0.3 if s <= r < t else 0.3
    if a.distributed:
        apex.parallel.convert_syncbn_model(j)
        j = DistributedDataParallel(j, delay_allreduce=True)
    u = torch.optim.SGD(j.parameters(), a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    v = torch.optim.lr_scheduler.CosineAnnealingLR(u, a.epochs, eta_min=1E-6)
    w = 0.
    x = 0.
    for y in range(a.epochs):
        z = a.drop_path_prob * y / a.epochs
        if a.distributed:
            j.module.drop_path_prob(z)
        else:
            j.drop_path_prob(z)
        if a.distributed:
            h.set_epoch(y)
        a(c, a, f, j, u, k, y, b)
        aa, ab = y(c, a, g, j, k, y, b)
        w = aa if aa > w else w
        x = ab if ab > x else x
        v.step()
    c.info("Final best Prec@1 = %.4f Prec@5 = %.4f", w, x)

if __name__ == "__main__":
    z()