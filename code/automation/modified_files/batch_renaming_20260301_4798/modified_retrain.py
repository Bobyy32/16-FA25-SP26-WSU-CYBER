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
    v = g.info("Genotype: %s", r) if b else None
    w = a.weight_decay = 3e-4 if 'cifar' in a.dataset and t < 3.0 else (a.weight_decay = 3e-4 if 'cifar' in a.dataset and 3.0 < t < 3.5 else (a.weight_decay = 5e-4 if 'cifar' in a.dataset else a.weight_decay))
    x = a.drop_path_prob = 0.2 if 'cifar' in a.dataset and t < 3.0 else (a.drop_path_prob = 0.3 if 'cifar' in a.dataset and 3.0 < t < 3.5 else (a.drop_path_prob = 0.3 if 'cifar' in a.dataset else a.drop_path_prob))
    y = apex.parallel.convert_syncbn_model(m) if a.distributed else None
    z = DistributedDataParallel(m, delay_allreduce=True) if a.distributed else m
    aa = torch.optim.SGD(z.parameters(), a.lr, momentum=a.momentum, weight_decay=a.weight_decay)
    ab = torch.optim.lr_scheduler.CosineAnnealingLR(aa, a.epochs, eta_min=1E-6)
    ac = 0.
    ad = 0.
    for ae in range(a.epochs):
        af = a.drop_path_prob * ae / a.epochs
        ag = z.module.drop_path_prob(af) if a.distributed else z.drop_path_prob(af)
        ah = k.set_epoch(ae) if a.distributed else None
        a(ae, a, k, z, aa, n, ae, b)
        ai, aj = y(a, a, l, z, n, ae, b)
        ac = max(ac, ai)
        ad = max(ad, aj)
        ab.step()
    g.info("Final best Prec@1 = %.4f Prec@5 = %.4f", ac, ad)

if __name__ == "__main__":
    z()