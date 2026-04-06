# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.data_utils import ImageNetPolicy
from datasets.data_utils import SubsetDistributedSampler


def build_imagenet_dataset(cfg):
    norml = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_path = os.path.join(cfg.data_dir, "train")
    test_path = os.path.join(cfg.data_dir, "val")
    
    if hasattr(cfg, "use_aa") and cfg.use_aa:
        training_data = dset.ImageFolder(
            train_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                norml,
            ]))
    else:
        training_data = dset.ImageFolder(
            train_path,
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

    testing_data = dset.ImageFolder(
        test_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norml,
        ]))

    return training_data, testing_data


def fetch_search_datasets(cfg):
    train_data, test_data = build_imagenet_dataset(cfg)
    total_count = len(train_data)
    indices_list = list(range(total_count))
    midpoint = int(np.floor(0.5 * total_count))

    if cfg.distributed:
        train_sample = SubsetDistributedSampler(train_data, indices_list[:midpoint])
        valid_sample = SubsetDistributedSampler(train_data, indices_list[midpoint:total_count])
    else:
        train_sample = torch.utils.data.sampler.SubsetRandomSampler(indices_list[:midpoint])
        valid_sample = torch.utils.data.sampler.SubsetRandomSampler(indices_list[midpoint:total_count])

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size,
        sampler=train_sample,
        pin_memory=True, num_workers=cfg.workers)

    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size,
        sampler=valid_sample,
        pin_memory=True, num_workers=cfg.workers)

    return [train_loader, valid_loader], [train_sample, valid_sample]


def fetch_augment_datasets(cfg):
    train_data, test_data = build_imagenet_dataset(cfg)
    
    if cfg.distributed:
        train_sample = torch.utils.data.distributed.DistributedSampler(train_data)
        test_sample = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sample = test_sample = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size,
        sampler=train_sample,
        pin_memory=True, num_workers=cfg.workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size,
        sampler=test_sample,
        pin_memory=True, num_workers=cfg.workers)

    return [train_loader, test_loader], [train_sample, test_sample]