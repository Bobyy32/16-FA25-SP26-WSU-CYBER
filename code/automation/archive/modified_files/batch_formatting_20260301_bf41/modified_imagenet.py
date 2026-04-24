# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from datasets.data_utils import ImageNetPolicy
from datasets.data_utils import SubsetDistributedSampler


def build_imagenet_dataset(config):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_directory = os.path.join(config.data_dir, "train")
    test_directory = os.path.join(config.data_dir, "val")
    
    if hasattr(config, "use_aa") and config.use_aa:
        training_dataset = dset.ImageFolder(
            train_directory,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                normalizer,
            ]))
    else:
        training_dataset = dset.ImageFolder(
            train_directory,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalizer,
            ]))

    validation_dataset = dset.ImageFolder(
        test_directory,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer,
        ]))

    return training_dataset, validation_dataset


def fetch_search_datasets(config):
    training_data, validation_data = build_imagenet_dataset(config)
    total_samples = len(training_data)
    indices_list = list(range(total_samples))
    midpoint = int(np.floor(0.5 * total_samples))

    if config.distributed:
        training_sampler = SubsetDistributedSampler(training_data, indices_list[:midpoint])
        validation_sampler = SubsetDistributedSampler(training_data, indices_list[midpoint:total_samples])
    else:
        training_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[:midpoint])
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_list[midpoint:total_samples])

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size,
        sampler=training_sampler,
        pin_memory=True, num_workers=config.workers)

    validation_loader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size,
        sampler=validation_sampler,
        pin_memory=True, num_workers=config.workers)

    return [training_loader, validation_loader], [training_sampler, validation_sampler]


def fetch_augment_datasets(config):
    training_data, validation_data = build_imagenet_dataset(config)
    
    if config.distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_data)
    else:
        training_sampler = validation_sampler = None

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=config.batch_size,
        sampler=training_sampler,
        pin_memory=True, num_workers=config.workers)

    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=config.batch_size,
        sampler=validation_sampler,
        pin_memory=True, num_workers=config.workers)

    return [training_loader, validation_loader], [training_sampler, validation_sampler]