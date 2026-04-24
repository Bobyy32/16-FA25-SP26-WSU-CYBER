# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.data_utils import ImageNetPolicy
from datasets.data_utils import SubsetDistributedSampler


def build_image_folder(cfg):
    norml = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn_path = os.path.join(cfg.data_dir, "train")
    tst_path = os.path.join(cfg.data_dir, "val")
    
    if hasattr(cfg, "use_aa") and cfg.use_aa:
        trn_dataset = dset.ImageFolder(
            trn_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                norml,
            ]))
    else:
        trn_dataset = dset.ImageFolder(
            trn_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                norml,
            ]))

    tst_dataset = dset.ImageFolder(
        tst_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norml,
        ]))

    return trn_dataset, tst_dataset


def fetch_search_datasets(cfg):
    trn_data, tst_data = build_image_folder(cfg)
    total_count = len(trn_data)
    indices_list = list(range(total_count))
    middle_point = int(np.floor(0.5 * total_count))

    if cfg.distributed:
        trn_sampler = SubsetDistributedSampler(trn_data, indices_list[:middle_point])
        val_sampler = SubsetDistributedSampler(trn_data, indices_list[middle_point:total_count])
    else:
        trn_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[:middle_point])
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[middle_point:total_count])

    trn_loader = torch.utils.data.DataLoader(
        trn_data, batch_size=cfg.batch_size,
        sampler=trn_sampler,
        pin_memory=True, num_workers=cfg.workers)

    val_loader = torch.utils.data.DataLoader(
        trn_data, batch_size=cfg.batch_size,
        sampler=val_sampler,
        pin_memory=True, num_workers=cfg.workers)

    return [trn_loader, val_loader], [trn_sampler, val_sampler]


def fetch_augment_datasets(cfg):
    trn_data, tst_data = build_image_folder(cfg)
    
    if cfg.distributed:
        trn_sampler = torch.utils.data.distributed.DistributedSampler(trn_data)
        tst_sampler = torch.utils.data.distributed.DistributedSampler(tst_data)
    else:
        trn_sampler = tst_sampler = None

    trn_loader = torch.utils.data.DataLoader(
        trn_data, batch_size=cfg.batch_size,
        sampler=trn_sampler,
        pin_memory=True, num_workers=cfg.workers)

    tst_loader = torch.utils.data.DataLoader(
        tst_data, batch_size=cfg.batch_size,
        sampler=tst_sampler,
        pin_memory=True, num_workers=cfg.workers)

    return [trn_loader, tst_loader], [trn_sampler, tst_sampler]