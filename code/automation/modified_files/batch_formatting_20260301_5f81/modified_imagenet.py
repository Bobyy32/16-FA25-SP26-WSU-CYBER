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
    normlz = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_path = os.path.join(cfg.data_dir, "train")
    test_path = os.path.join(cfg.data_dir, "val")
    
    if hasattr(cfg, "use_aa") and cfg.use_aa:
        training_set = dset.ImageFolder(
            train_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normlz,
            ]))
    else:
        training_set = dset.ImageFolder(
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
                normlz,
            ]))

    validation_set = dset.ImageFolder(
        test_path,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normlz,
        ]))

    return training_set, validation_set


def fetch_search_datasets(cfg):
    training_data, validation_data = build_imagenet_dataset(cfg)
    total_count = len(training_data)
    indices_list = list(range(total_count))
    midpoint = int(np.floor(0.5 * total_count))

    if cfg.distributed:
        train_sampler = SubsetDistributedSampler(training_data, indices_list[:midpoint])
        valid_sampler = SubsetDistributedSampler(training_data, indices_list[midpoint:total_count])
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[:midpoint])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[midpoint:total_count])

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=cfg.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=cfg.workers)

    valid_loader = torch.utils.data.DataLoader(
        training_data, batch_size=cfg.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=cfg.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def fetch_augment_datasets(cfg):
    training_data, validation_data = build_imagenet_dataset(cfg)
    
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(validation_data)
    else:
        train_sampler = test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=cfg.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=cfg.workers)

    test_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=cfg.batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=cfg.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]