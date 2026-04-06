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

def y(b, c, d, e, f, g, h, i):
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

def z():
    a = RetrainConfig()
    b = not a.distributed or a.local_rank == 0
    c = torch.cuda.set_device(a.local_rank) if a.distributed else None
    d = torch.distributed.init_process_group(backend='nccl', init_method=a.dist_url, rank=a.local_rank, world_size=a.world_size) if a.distributed else None
    e = os.makedirs(a.output_path, exist_ok=True) if b else None
    f = torch.distributed.barrier() if a.distributed else None
    g = utils.get_logger(os.path.join(a.output_path, 'search.log'))
    h = a.print_params(g.info) if b else None
    utils.reset_seed(a.seed)
    i, j = get_augment_datasets(a)
    k, l = i, j
    m = Model(a.dataset, a.layers, in_channels=a.input_channels, channels=a.init_channels, retrain=True).cuda()
    n = utils.CrossEntropyLabelSmooth(a.n_classes, a.label_smooth) if a.label_smooth > 0 else nn.CrossEntropyLoss()
    o = os.path.join(a.output_path, a.arc_checkpoint)
    with open(o, "r") as p:
        q = json.load(p)
    r = utils.encode_tensor(q, torch.device("cuda"))
    s = utils.parse_results(r, n_nodes=4)
    t = {u: s for u in range(3)}
    apply_fixed_architecture(m, o)
    v = utils.param_size(m, n, [3, 32, 32] if 'cifar' in a.dataset else [3, 224, 224])
    w = g.info("Param size: %.6f", v) if b else None
    x = g.info("Genotype: %s", s) if b else None
    y = 3.0 if 'cifar' in a.dataset else None
    z = 3.0 if 'cifar' in a.dataset else None
    a.weight_decay = 3e-4 if y < 3.0 else 5e-4 if z >= 3.5 else 3e-4
    a.drop_path_prob = 0.2 if y < 3.0 else 0.3 if z >= 3.5 else 0.3
    aa = apex.parallel.convert_syncbn_model(m) if a.distributed else None
    ab = DistributedDataParallel(m, delay_allreduce=True) if a.distributed else m
    ac = torch.optim.SGD(ab.parameters(), a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    ad = torch.optim.lr_scheduler.CosineAnnealingLR(ac, a.epochs, eta_min=1E-6)
    ae = 0.
    af = 0.
    for ag in range(a.epochs):
        ah = a.drop_path_prob * ag / a.epochs
        ai = ab.module.drop_path_prob(ah) if a.distributed else ab.drop_path_prob(ah)
        aj = k.set_epoch(ag) if a.distributed else None
        a(b, a, k, ab, ac, n, ag, b)
        ak, al = y(b, a, l, ab, n, ag, b)
        ae = max(ae, ak)
        af = max(af, al)
        ad.step()
    b.info("Final best Prec@1 = %.4f Prec@5 = %.4f", ae, af)

if __name__ == "__main__":
    z()