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
    m = b.info("Epoch %d LR %.6f", i, l) if j else None
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
        w = j and (n % g.log_frequency == 0 or n + 1 == len(d))
        x = b.info("Epoch [%d/%d] Step [%d/%d]  %s", i + 1, g.epochs, n + 1, len(d), k) if w else None
    y = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", i + 1, g.epochs, k.prec1.avg, k.prec5.avg) if j else None

def z(b, c, d, e, f, g, h):
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
            v = h and (n % g.log_frequency == 0 or n + 1 == len(d))
            w = b.info("Epoch [%d/%d] Step [%d/%d]  %s", g.epochs, g.epochs, n + 1, len(d), k) if v else None
    x = b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", g.epochs, g.epochs, k.prec1.avg, k.prec5.avg) if h else None
    return k.prec1.avg, k.prec5.avg

def y():
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
    q = utils.encode_tensor(q, torch.device("cuda"))
    r = utils.parse_results(q, n_nodes=4)
    s = {i: r for i in range(3)}
    apply_fixed_architecture(m, o)
    t = utils.param_size(m, n, [3, 32, 32] if 'cifar' in a.dataset else [3, 224, 224])
    u = g.info("Param size: %.6f", t) if b else None
    u = g.info("Genotype: %s", r) if b else None
    v = a.weight_decay = 3e-4 if 'cifar' in a.dataset and t < 3.0 else (a.weight_decay = 3e-4 if 'cifar' in a.dataset and 3.0 < t < 3.5 else (a.weight_decay = 5e-4 if 'cifar' in a.dataset else None))
    w = a.drop_path_prob = 0.2 if 'cifar' in a.dataset and t < 3.0 else (a.drop_path_prob = 0.3 if 'cifar' in a.dataset and 3.0 < t < 3.5 else (a.drop_path_prob = 0.3 if 'cifar' in a.dataset else None))
    x = apex.parallel.convert_syncbn_model(m) if a.distributed else None
    y = DistributedDataParallel(m, delay_allreduce=True) if a.distributed else None
    z = torch.optim.SGD(m.parameters(), a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    aa = torch.optim.lr_scheduler.CosineAnnealingLR(z, a.epochs, eta_min=1E-6)
    ab = 0.
    ac = 0.
    for ad in range(a.epochs):
        ae = a.drop_path_prob * ad / a.epochs
        af = m.module.drop_path_prob(ae) if a.distributed else m.drop_path_prob(ae)
        ag = k.set_epoch(ad) if a.distributed else None
        a(a, z, k, m, z, n, ad, b)
        ah, ai = z(a, z, l, m, n, ad, b)
        ab = ab if ab > ah else ah
        ac = ac if ac > ai else ai
        aa.step()
    a.info("Final best Prec@1 = %.4f Prec@5 = %.4f", ab, ac)

if __name__ == "__main__":
    y()