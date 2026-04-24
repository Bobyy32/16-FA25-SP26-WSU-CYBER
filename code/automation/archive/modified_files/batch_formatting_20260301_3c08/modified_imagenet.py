# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import path
from numpy import floor
from torch import utils
from torchvision import datasets as dset, transforms
from datasets.data_utils import ImageNetPolicy, SubsetDistributedSampler


def _imagenet_dataset(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dir = path.join(config.data_dir, "train")
    test_dir = path.join(config.data_dir, "val")
    if hasattr(config, "use_aa") and config.use_aa:
        train_data = dset.ImageFolder(train_dir, transforms.Compose([transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), ImageNetPolicy(), transforms.ToTensor(), normalize, ]))
    else:
        train_data = dset.ImageFolder(train_dir, transforms.Compose([transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
            hue=0.2), transforms.ToTensor(), normalize, ]))

    test_data = dset.ImageFolder(test_dir, transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize, ]))
    return train_data, test_data


def get_search_datasets(config):
    train_data, test_data = _imagenet_dataset(config)
    num_train = len(train_data)
    indices = list(range(num_train))
    split_mid = int(floor(0.5 * num_train))
    if config.distributed:
        train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
        valid_sampler = SubsetDistributedSampler(train_data, indices[split_mid:num_train])
    else:
        train_sampler = utils.data.sampler.SubsetRandomSampler(indices[:split_mid])
        valid_sampler = utils.data.sampler.SubsetRandomSampler(indices[split_mid:num_train])

    train_loader = utils.data.DataLoader(train_data, batch_size=config.batch_size,
        sampler=train_sampler, pin_memory=True, num_workers=config.workers)

    valid_loader = utils.data.DataLoader(train_data, batch_size=config.batch_size,
        sampler=valid_sampler, pin_memory=True, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]


def get_augment_datasets(config):
    train_data, test_data = _imagenet_dataset(config)
    if config.distributed:
        train_sampler = utils.data.distributed.DistributedSampler(train_data)
        test_sampler = utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = test_sampler = None

    train_loader = utils.data.DataLoader(train_data, batch_size=config.batch_size,
        sampler=train_sampler, pin_memory=True, num_workers=config.workers)

    test_loader = utils.data.DataLoader(test_data, batch_size=config.batch_size,
        sampler=test_sampler, pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]