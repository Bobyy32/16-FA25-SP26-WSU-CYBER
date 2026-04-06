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

def a(b,c,d,e,f,g,h):
    i=AverageMeterGroup()
    j=f.param_groups[0]["lr"]
    if h:
        b.info("Epoch %d LR %.6f", e, j)
    f.train()
    for k,(l,m) in enumerate(d):
        l,m=l.cuda(non_blocking=True),m.cuda(non_blocking=True)
        f.zero_grad()
        n,o=g(l)
        p=c(n,m)
        if f.aux_weight>0.:
            p+=f.aux_weight*c(o,m)
        p.backward()
        nn.utils.clip_grad_norm_(g.parameters(), f.grad_clip)
        f.step()
        q,r=utils.accuracy(n,m,topk=(1, 5))
        s={"prec1":q,"prec5":r,"loss":p}
        s=utils.reduce_metrics(s,f.distributed)
        i.update(s)
        if h and (k%f.log_frequency==0 or k+1==len(d)):
            b.info("Epoch [%d/%d] Step [%d/%d]  %s", e+1,f.epochs,k+1,len(d),i)
    if h:
        b.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", e+1,f.epochs,i.prec1.avg,i.prec5.avg)

def t(u,v,w,x,y,z):
    i=AverageMeterGroup()
    x.eval()
    with torch.no_grad():
        for k,(l,m) in enumerate(w):
            l,m=l.cuda(non_blocking=True),m.cuda(non_blocking=True)
            n,_=x(l)
            p=c(n,m)
            q,r=utils.accuracy(n,m,topk=(1, 5))
            s={"prec1":q,"prec5":r,"loss":p}
            s=utils.reduce_metrics(s,y.distributed)
            i.update(s)
            if z and (k%y.log_frequency==0 or k+1==len(w)):
                u.info("Epoch [%d/%d] Step [%d/%d]  %s", y.epochs,k+1,len(w),y.epochs,i)
    if z:
        u.info("Train: [%d/%d] Final Prec@1 %.4f Prec@5 %.4f", y.epochs,i.prec1.avg,i.prec5.avg)
    return i.prec1.avg,i.prec5.avg

def m():
    f=RetrainConfig()
    h=not f.distributed or f.local_rank==0
    if f.distributed:
        torch.cuda.set_device(f.local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method=f.dist_url,rank=f.local_rank,world_size=f.world_size)
    if h:
        os.makedirs(f.output_path,exist_ok=True)
    if f.distributed:
        torch.distributed.barrier()
    b=utils.get_logger(os.path.join(f.output_path,'search.log'))
    if h:
        f.print_params(b.info)
    utils.reset_seed(f.seed)
    o,p=get_augment_datasets(f)
    q,r=o
    s,t=p
    g=Model(f.dataset,f.layers,in_channels=f.input_channels,channels=f.init_channels,retrain=True).cuda()
    if f.label_smooth>0:
        c=utils.CrossEntropyLabelSmooth(f.n_classes,f.label_smooth)
    else:
        c=nn.CrossEntropyLoss()
    u=os.path.join(f.output_path,f.arc_checkpoint)
    with open(u,"r") as v:
        w=json.load(v)
    w=utils.encode_tensor(w,torch.device("cuda"))
    x=utils.parse_results(w,n_nodes=4)
    y={i:x for i in range(3)}
    apply_fixed_architecture(g,u)
    z=utils.param_size(g,c,[3,32,32] if 'cifar' in f.dataset else [3,224,224])
    if h:
        b.info("Param size: %.6f", z)
        b.info("Genotype: %s", x)
    z='cifar' in f.dataset
    f.weight_decay=3e-4 if z and z<3.0 else 3e-4 if z and 3.0<z<3.5 else 5e-4
    f.drop_path_prob=0.2 if z and z<3.0 else 0.3 if z and 3.0<z<3.5 else 0.3
    if f.distributed:
        apex.parallel.convert_syncbn_model(g)
        g=DistributedDataParallel(g,delay_allreduce=True)
    d=torch.optim.SGD(g.parameters(),f.lr,momentum=f.momentum,weight_decay=f.weight_decay)
    e=torch.optim.lr_scheduler.CosineAnnealingLR(d,f.epochs,eta_min=1E-6)
    l=m=0.
    for n in range(f.epochs):
        o=f.drop_path_prob*n/f.epochs
        if f.distributed:
            g.module.drop_path_prob(o)
        else:
            g.drop_path_prob(o)
        if f.distributed:
            s.set_epoch(n)
        a(b,f,q,g,d,c,n,h)
        k,w=t(b,f,r,g,c,n,h)
        l=max(l,k)
        m=max(m,w)
        e.step()
    b.info("Final best Prec@1 = %.4f Prec@5 = %.4f", l, m)

if __name__ == "__main__":
    m()