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

def y(z, A, B, C, D, E, F):
    G = AverageMeterGroup()
    C.eval()
    with torch.no_grad():
        for H, (I, J) in enumerate(B):
            I, J = I.cuda(non_blocking=True), J.cuda(non_blocking=True)
            K, _ = C(I)
            L = D(K, J)
            M, N = utils.accuracy(K, J, topk=(1, 5))
            O = {"prec1": M, "prec5": N, "loss": L}
            O = utils.reduce_metrics(O, A.distributed)
            G.update(O)
            P = F and (H % A.log_frequency == 0 or H + 1 == len(B))
            Q = z.info("Epoch [%d/%d] Step [%d/%d]  %s", E + 1, A.epochs, H + 1, len(B), G) if P else None
    R = z.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", E + 1, A.epochs, G.prec1.avg, G.prec5.avg) if F else None
    return G.prec1.avg, G.prec5.avg

def S():
    T = RetrainConfig()
    U = not T.distributed or T.local_rank == 0
    if T.distributed:
        torch.cuda.set_device(T.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=T.dist_url, rank=T.local_rank, world_size=T.world_size)
    V = os.makedirs(T.output_path, exist_ok=True) if U else None
    if T.distributed:
        torch.distributed.barrier()
    W = utils.get_logger(os.path.join(T.output_path, 'search.log'))
    X = T.print_params(W.info) if U else None
    utils.reset_seed(T.seed)
    Y, Z = get_augment_datasets(T)
    _, _ = Y
    _, _ = Z
    _ = Model(T.dataset, T.layers, in_channels=T.input_channels, channels=T.init_channels, retrain=True).cuda()
    _ = utils.CrossEntropyLabelSmooth(T.n_classes, T.label_smooth) if T.label_smooth > 0 else nn.CrossEntropyLoss()
    _ = os.path.join(T.output_path, T.arc_checkpoint)
    with open(_, "r") as _:
        _ = json.load(_)
    _ = utils.encode_tensor(_, torch.device("cuda"))
    _ = utils.parse_results(_, n_nodes=4)
    _ = {i: _ for i in range(3)}
    apply_fixed_architecture(_, _)
    _ = utils.param_size(_, _, [3, 32, 32] if 'cifar' in T.dataset else [3, 224, 224])
    _ = W.info("Param size: %.6f", _) if U else None
    _ = W.info("Genotype: %s", _) if U else None
    _ = 'cifar' in T.dataset
    _ = _ and (_ < 3.0 and (T.weight_decay = 3e-4, T.drop_path_prob = 0.2) or (3.0 < _ < 3.5 and (T.weight_decay = 3e-4, T.drop_path_prob = 0.3) or (T.weight_decay = 5e-4, T.drop_path_prob = 0.3)))
    if T.distributed:
        apex.parallel.convert_syncbn_model(_)
        _ = DistributedDataParallel(_, delay_allreduce=True)
    _ = torch.optim.SGD(_.parameters(), T.lr, momentum=T.momentum, weight_decay=T.weight_decay)
    _ = torch.optim.lr_scheduler.CosineAnnealingLR(_, T.epochs, eta_min=1E-6)
    _ = 0.
    _ = 0.
    for _ in range(T.epochs):
        _ = T.drop_path_prob * _ / T.epochs
        _ = _.module.drop_path_prob(_) if T.distributed else _.drop_path_prob(_)
        _ = T.distributed and _.set_epoch(_) or None
        a(W, _, Y, _, _, T, U, _)
        _, _ = y(W, T, B, _, _, _, U)
        _ = max(_, _)
        _ = max(_, _)
        _.step()
    W.info("Final best Prec@1 = %.4f Prec@5 = %.4f", _, _)

if __name__ == "__main__":
    S()