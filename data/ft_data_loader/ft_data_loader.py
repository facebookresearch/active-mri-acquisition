"""
Source : https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py

Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
import os
import PIL
# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .ft_cifar10 import FT_CIFAR10
from .ft_imagenet import FT_ImageNet
from .ft_mnist import FT_MNIST


def get_norm_transform(normalize):
    if normalize == 'gan':
        normalize_tf = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    elif normalize == 'zero_one':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
        )
    elif normalize == 'cae':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[1, 1, 1],
        )
    elif normalize == 'imagenet':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0.43, 0.43, 0.43],
            std=[0.23, 0.23, 0.23],
        )
    return normalize_tf

# if fine_size is 128, load_size can be 144
def get_train_valid_loader(batch_size,
                           load_size, 
                           fine_size,
                           keep_ratio,
                           augment,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           normalize='gan',
                           which_dataset='MNIST',
                           data_dir='/private/home/zizhao/work/data/'
                         ):
    random_seed = 1234
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize_tf = get_norm_transform(normalize)
    # define transforms
    valid_transform = transforms.Compose([
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(fine_size),
            transforms.ToTensor(),
            normalize_tf,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.RandomCrop(fine_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_tf,
        ])
    else:
        train_transform = valid_transform

    print('load {} train/val (val ratio {:.4f}) dataset'.format(which_dataset, valid_size))

    if which_dataset == 'CIFAR10':
        dataset = FT_CIFAR10
        data_dir = '/private/home/zizhao/work/data/'
    elif which_dataset == 'ImageNet':
        dataset = FT_ImageNet
        data_dir = '/datasets01/imagenet_resized_144px/060718/061417'
    elif which_dataset == 'MNIST':
        dataset = FT_MNIST

    # load the dataset
    dataset = dataset(
        root=data_dir, train=True, normalize=normalize,
        download=True, transform=train_transform,  unmask_ratio=keep_ratio,
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
        )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
 
    return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    load_size,
                    fine_size,
                    keep_ratio,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=False,
                    normalize='gan',
                    which_dataset='MNIST',
                    data_dir='/private/home/zizhao/work/data/'
                    ):

    normalize_tf = get_norm_transform(normalize)
    # define transform
    transform = transforms.Compose([
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(fine_size),
            transforms.ToTensor(),
            normalize_tf,
    ])

    print('load {} test dataset'.format(which_dataset))

    if which_dataset == 'CIFAR10':
        dataset = FT_CIFAR10
    elif which_dataset == 'ImageNet':
        dataset = FT_ImageNet
        data_dir = '/datasets01/imagenet_resized_144px/060718/061417'
    elif which_dataset == 'MNIST':
        dataset = FT_MNIST

    dataset = dataset(
        root=data_dir, train=False, normalize=normalize,
        transform=transform,  unmask_ratio=keep_ratio,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader

