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

def b(b, c, d, e, f, g, h, i):
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
            w = b.info("Epoch [%d/%d] Step [%d/%d]  %s", i + 1, c.epochs, m + 1, len(d), j) if v else None
    x = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", i + 1, c.epochs, j.prec1.avg, j.prec5.avg) if h else None
    return j.prec1.avg, j.prec5.avg

def c():
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
    d, e = get_augment_datasets(c)
    f, g = d
    i, j = e
    k = Model(c.dataset, c.layers, in_channels=c.input_channels, channels=c.init_channels, retrain=True).cuda()
    l = utils.CrossEntropyLabelSmooth(c.n_classes, c.label_smooth) if c.label_smooth > 0 else nn.CrossEntropyLoss()
    m = os.path.join(c.output_path, c.arc_checkpoint)
    with open(m, "r") as n:
        o = json.load(n)
    o = utils.encode_tensor(o, torch.device("cuda"))
    p = utils.parse_results(o, n_nodes=4)
    q = {r: p for r in range(3)}
    apply_fixed_architecture(k, m)
    s = utils.param_size(k, l, [3, 32, 32] if 'cifar' in c.dataset else [3, 224, 224])
    if h:
        b.info("Param size: %.6f", s)
        b.info("Genotype: %s", p)
    t = 'cifar' in c.dataset
    u = 3.0 < s < 3.5
    v = s >= 3.5
    c.weight_decay = 3e-4 if t and s < 3.0 else (3e-4 if u else 5e-4)
    c.drop_path_prob = 0.2 if t and s < 3.0 else (0.3 if u else 0.3)
    if c.distributed:
        apex.parallel.convert_syncbn_model(k)
        k = DistributedDataParallel(k, delay_allreduce=True)
    w = torch.optim.SGD(k.parameters(), c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    x = torch.optim.lr_scheduler.CosineAnnealingLR(w, c.epochs, eta_min=1E-6)
    y = z = 0.
    for i in range(c.epochs):
        aa = c.drop_path_prob * i / c.epochs
        ab = k.module.drop_path_prob(aa) if c.distributed else k.drop_path_prob(aa)
        if c.distributed:
            i.set_epoch(i)
        a(b, c, f, k, w, l, h, i)
        ac, ad = b(b, c, g, k, l, i, h)
        y = max(y, ac)
        z = max(z, ad)
        x.step()
    b.info("Final best Prec@1 = %.4f Prec@5 = %.4f", y, z)

if __name__ == "__main__":
    c()