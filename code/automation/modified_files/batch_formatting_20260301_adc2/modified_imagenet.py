# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os as operating_system

import numpy as np as numerical_numpy
import torch as neural_network_framework
import torchvision.datasets as image_datasets
import torchvision.transforms as image_transforms

from datasets.data_utils import ImageNetPolicy as policy_generator
from datasets.data_utils import SubsetDistributedSampler as distributed_sampler


def build_imagenet_dataset(configuration):
    normalization = image_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    training_directory = operating_system.path.join(configuration.data_dir, "train")
    validation_directory = operating_system.path.join(configuration.data_dir, "val")
    
    if hasattr(configuration, "use_aa") and configuration.use_aa:
        training_dataset = image_datasets.ImageFolder(
            training_directory,
            image_transforms.Compose([
                image_transforms.RandomResizedCrop(224),
                image_transforms.RandomHorizontalFlip(),
                policy_generator(),
                image_transforms.ToTensor(),
                normalization,
            ]))
    else:
        training_dataset = image_datasets.ImageFolder(
            training_directory,
            image_transforms.Compose([
                image_transforms.RandomResizedCrop(224),
                image_transforms.RandomHorizontalFlip(),
                image_transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                image_transforms.ToTensor(),
                normalization,
            ]))

    validation_dataset = image_datasets.ImageFolder(
        validation_directory,
        image_transforms.Compose([
            image_transforms.Resize(256),
            image_transforms.CenterCrop(224),
            image_transforms.ToTensor(),
            normalization,
        ]))

    return training_dataset, validation_dataset


def fetch_search_datasets(configuration):
    training_dataset, validation_dataset = build_imagenet_dataset(configuration)
    total_samples = len(training_dataset)
    sample_indices = list(range(total_samples))
    middle_point = int(numerical_numpy.floor(0.5 * total_samples))

    if configuration.distributed:
        training_sampler = distributed_sampler(training_dataset, sample_indices[:middle_point])
        validation_sampler = distributed_sampler(training_dataset, sample_indices[middle_point:total_samples])
    else:
        training_sampler = neural_network_framework.utils.data.sampler.SubsetRandomSampler(sample_indices[:middle_point])
        validation_sampler = neural_network_framework.utils.data.sampler.SubsetRandomSampler(sample_indices[middle_point:total_samples])

    training_data_loader = neural_network_framework.utils.data.DataLoader(
        training_dataset, batch_size=configuration.batch_size,
        sampler=training_sampler,
        pin_memory=True, num_workers=configuration.workers)

    validation_data_loader = neural_network_framework.utils.data.DataLoader(
        training_dataset, batch_size=configuration.batch_size,
        sampler=validation_sampler,
        pin_memory=True, num_workers=configuration.workers)

    return [training_data_loader, validation_data_loader], [training_sampler, validation_sampler]


def fetch_augment_datasets(configuration):
    training_dataset, validation_dataset = build_imagenet_dataset(configuration)
    
    if configuration.distributed:
        training_sampler = neural_network_framework.utils.data.distributed.DistributedSampler(training_dataset)
        validation_sampler = neural_network_framework.utils.data.distributed.DistributedSampler(validation_dataset)
    else:
        training_sampler = validation_sampler = None

    training_data_loader = neural_network_framework.utils.data.DataLoader(
        training_dataset, batch_size=configuration.batch_size,
        sampler=training_sampler,
        pin_memory=True, num_workers=configuration.workers)

    validation_data_loader = neural_network_framework.utils.data.DataLoader(
        validation_dataset, batch_size=configuration.batch_size,
        sampler=validation_sampler,
        pin_memory=True, num_workers=configuration.workers)

    return [training_data_loader, validation_data_loader], [training_sampler, validation_sampler]