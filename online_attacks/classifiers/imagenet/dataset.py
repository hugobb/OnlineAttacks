# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from online_attacks.classifiers.dataset import DatasetParams
import os
import ipdb

def load_imagenet_dataset(
    params: DatasetParams = DatasetParams(), train: bool = True
):
    data_root = '/datasets01/imagenet_full_size/061417/'
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    testdir = os.path.join(data_root, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if train:
        imagenet_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize, ]))
    else:
        imagenet_dataset =  datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256), transforms.CenterCrop(224),
            transforms.ToTensor(), normalize, ]))
    return imagenet_dataset

def create_imagenet_loaders(
    params: DatasetParams = DatasetParams()
) -> (DataLoader, DataLoader):

    # Data loading code
    data_root = '/datasets01/imagenet_full_size/061417/'
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    testdir = os.path.join(data_root, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=params.batch_size, shuffle=True,
        num_workers=params.num_workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=params.test_batch_size, shuffle=False,
        num_workers=params.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=params.test_batch_size, shuffle=False,
        num_workers=params.num_workers, pin_memory=True)

    return train_loader, test_loader
