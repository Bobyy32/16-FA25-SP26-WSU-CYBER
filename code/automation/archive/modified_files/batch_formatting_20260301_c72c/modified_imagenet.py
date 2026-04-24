# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os as os_path
import numpy as np_array
import torch as torch_lib
import torchvision.datasets as dataset_lib
import torchvision.transforms as transform_lib

from datasets.data_utils import ImageNetPolicy as policy_class
from datasets.data_utils import SubsetDistributedSampler as sampler_class


def build_imagenet_dataset(config_obj):
    norm_transform = transform_lib.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_path = os_path.path.join(config_obj.data_dir, "train")
    test_path = os_path.path.join(config_obj.data_dir, "val")
    
    if hasattr(config_obj, "use_aa") and config_obj.use_aa:
        train_dataset = dataset_lib.ImageFolder(
            train_path,
            transform_lib.Compose([
                transform_lib.RandomResizedCrop(224),
                transform_lib.RandomHorizontalFlip(),
                policy_class(),
                transform_lib.ToTensor(),
                norm_transform,
            ]))
    else:
        train_dataset = dataset_lib.ImageFolder(
            train_path,
            transform_lib.Compose([
                transform_lib.RandomResizedCrop(224),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transform_lib.ToTensor(),
                norm_transform,
            ]))

    test_dataset = dataset_lib.ImageFolder(
        test_path,
        transform_lib.Compose([
            transform_lib.Resize(256),
            transform_lib.CenterCrop(224),
            transform_lib.ToTensor(),
            norm_transform,
        ]))

    return train_dataset, test_dataset


def fetch_search_datasets(config_obj):
    train_dataset, test_dataset = build_imagenet_dataset(config_obj)
    total_samples = len(train_dataset)
    sample_indices = list(range(total_samples))
    midpoint = int(np_array.floor(0.5 * total_samples))

    if config_obj.distributed:
        train_sampler = sampler_class(train_dataset, sample_indices[:midpoint])
        valid_sampler = sampler_class(train_dataset, sample_indices[midpoint:total_samples])
    else:
        train_sampler = torch_lib.utils.data.sampler.SubsetRandomSampler(sample_indices[:midpoint])
        valid_sampler = torch_lib.utils.data.sampler.SubsetRandomSampler(sample_indices[midpoint:total_samples])

    train_loader = torch_lib.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    valid_loader = torch_lib.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=valid_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def fetch_augment_datasets(config_obj):
    train_dataset, test_dataset = build_imagenet_dataset(config_obj)
    
    if config_obj.distributed:
        train_sampler = torch_lib.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch_lib.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = test_sampler = None

    train_loader = torch_lib.utils.data.DataLoader(
        train_dataset, batch_size=config_obj.batch_size,
        sampler=train_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    test_loader = torch_lib.utils.data.DataLoader(
        test_dataset, batch_size=config_obj.batch_size,
        sampler=test_sampler,
        pin_memory=True, num_workers=config_obj.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]