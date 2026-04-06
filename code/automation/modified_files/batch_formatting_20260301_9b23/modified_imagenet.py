# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os as os_module

import numpy as np as numpy_module
import torch as torch_module
import torchvision.datasets as datasets_module
import torchvision.transforms as transforms_module

from datasets.data_utils import ImageNetPolicy as policy_class
from datasets.data_utils import SubsetDistributedSampler as sampler_class


def build_imagenet_dataset(config_obj):
    norm_transform = transforms_module.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_path = os_module.path.join(config_obj.data_dir, "train")
    test_path = os_module.path.join(config_obj.data_dir, "val")
    
    if hasattr(config_obj, "use_aa") and config_obj.use_aa:
        training_dataset = datasets_module.ImageFolder(
            train_path,
            transforms_module.Compose([
                transforms_module.RandomResizedCrop(224),
                transforms_module.RandomHorizontalFlip(),
                policy_class(),
                transforms_module.ToTensor(),
                norm_transform,
            ]))
    else:
        training_dataset = datasets_module.ImageFolder(
            train_path,
            transforms_module.Compose([
                transforms_module.RandomResizedCrop(224),
                transforms_module.RandomHorizontalFlip(),
                transforms_module.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms_module.ToTensor(),
                norm_transform,
            ]))

    validation_dataset = datasets_module.ImageFolder(
        test_path,
        transforms_module.Compose([
            transforms_module.Resize(256),
            transforms_module.CenterCrop(224),
            transforms_module.ToTensor(),
            norm_transform,
        ]))

    return training_dataset, validation_dataset


def fetch_search_datasets(config_obj):
    train_dataset, valid_dataset = build_imagenet_dataset(config_obj)
    total_samples = len(train_dataset)
    sample_indices = list(range(total_samples))
    midpoint = int(numpy_module.floor(0.5 * total_samples))

    if config_obj.distributed:
        train_sampler = sampler_class(train_dataset, sample_indices[:midpoint])
        valid_sampler = sampler_class(train_dataset, sample_indices[midpoint:total_samples])
    else:
        train_sampler = torch_module.utils.data.sampler.SubsetRandomSampler(sample_indices[:midpoint])
        valid_sampler = torch_module.utils.data.sampler.SubsetRandomSampler(sample_indices[midpoint:total_samples])

    train_loader = torch_module.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    valid_loader = torch_module.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def fetch_augment_datasets(config_obj):
    train_dataset, valid_dataset = build_imagenet_dataset(config_obj)
    
    if config_obj.distributed:
        train_sampler = torch_module.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch_module.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = valid_sampler = None

    train_loader = torch_module.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    valid_loader = torch_module.utils.data.DataLoader(
        valid_dataset, batch_size=config_obj.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]