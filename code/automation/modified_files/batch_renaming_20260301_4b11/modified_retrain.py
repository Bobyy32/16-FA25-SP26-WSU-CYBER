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

def y(b, c, d, e, f, g, h):
    j = AverageMeterGroup()
    e.eval()
    with torch.no_grad():
        for m, (n, o) in enumerate(d):
            n, o = n.cuda(non_blocking=True), o.cuda(non_blocking=True)
            p, _ = e(n)
            r = c(p, o)
            s, t = utils.accuracy(p, o, topk=(1, 5))
            u = {"prec1": s, "prec5": t, "loss": r}
            u = utils.reduce_metrics(u, g.distributed)
            j.update(u)
            v = h and (m % g.log_frequency == 0 or m + 1 == len(d))
            w = b.info("Epoch [%d/%d] Step [%d/%d]  %s", g.epochs, g.epochs, m + 1, len(d), j) if v else None
    x = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", g.epochs, g.epochs, j.prec1.avg, j.prec5.avg) if h else None
    return j.prec1.avg, j.prec5.avg

def z():
    a = RetrainConfig()
    b = not a.distributed or a.local_rank == 0
    if a.distributed:
        torch.cuda.set_device(a.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=a.dist_url, rank=a.local_rank, world_size=a.world_size)
    c = os.makedirs(a.output_path, exist_ok=True) if b else None
    if a.distributed:
        torch.distributed.barrier()
    d = utils.get_logger(os.path.join(a.output_path, 'search.log'))
    if b:
        a.print_params(d.info)
    utils.reset_seed(a.seed)
    e, f = get_augment_datasets(a)
    g, h = e
    i, j = f
    k = Model(a.dataset, a.layers, in_channels=a.input_channels, channels=a.init_channels, retrain=True).cuda()
    l = utils.CrossEntropyLabelSmooth(a.n_classes, a.label_smooth) if a.label_smooth > 0 else nn.CrossEntropyLoss()
    m = os.path.join(a.output_path, a.arc_checkpoint)
    with open(m, "r") as n:
        o = json.load(n)
    o = utils.encode_tensor(o, torch.device("cuda"))
    p = utils.parse_results(o, n_nodes=4)
    q = {r: p for r in range(3)}
    apply_fixed_architecture(k, m)
    s = utils.param_size(k, l, [3, 32, 32] if 'cifar' in a.dataset else [3, 224, 224])
    if b:
        d.info("Param size: %.6f", s)
        d.info("Genotype: %s", p)
    t = 3.0 if 'cifar' in a.dataset else 0
    u = 3.0 if 'cifar' in a.dataset else 0
    v = 3.5 if 'cifar' in a.dataset else 0
    a.weight_decay = 3e-4 if t < s < u else (5e-4 if s >= v else 3e-4)
    a.drop_path_prob = 0.2 if t < s < u else (0.3 if s >= v else 0.3)
    if a.distributed:
        apex.parallel.convert_syncbn_model(k)
        k = DistributedDataParallel(k, delay_allreduce=True)
    w = torch.optim.SGD(k.parameters(), a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    x = torch.optim.lr_scheduler.CosineAnnealingLR(w, a.epochs, eta_min=1E-6)
    y = z = 0.
    for i in range(a.epochs):
        aa = a.drop_path_prob * i / a.epochs
        ab = k.module.drop_path_prob(aa) if a.distributed else k.drop_path_prob(aa)
        if a.distributed:
            i.set_epoch(i)
        a(d, a, g, k, w, l, i, b)
        ac, ad = y(d, a, h, k, l, i, b)
        y = ac if ac > y else y
        z = ad if ad > z else z
        x.step()
    d.info("Final best Prec@1 = %.4f Prec@5 = %.4f", y, z)

if __name__ == "__main__":
    z()