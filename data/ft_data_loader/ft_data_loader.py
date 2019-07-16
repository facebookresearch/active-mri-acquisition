"""
Source : https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py

Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
import os, sys
import pathlib
import PIL

# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from .ft_cifar10 import FT_CIFAR10
from .ft_imagenet import FT_ImageNet
from .ft_mnist import FT_MNIST
from .ft_util_vaes import MaskFunc, DicomDataTransform, Slice, FixedAccelerationMaskFunc, RawDataTransform, \
    RawSliceData, FixedOrderRandomSampler
import random


def get_train_valid_loader(batch_size, num_workers=4, pin_memory=False, which_dataset='KNEE'):

    if which_dataset == 'KNEE':
        mask_func = FixedAccelerationMaskFunc([0.125], [4])
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform = DicomDataTransform(mask_func, None)
        train_data = Slice(
            data_transform,
            dicom_root,
            which='train',
            resolution=128,
            scan_type='all',
            num_volumes=None,
            num_rand_slices=None)
        valid_data = Slice(
            data_transform,
            dicom_root,
            which='val',
            resolution=128,
            scan_type='all',
            num_volumes=None,
            num_rand_slices=None)

        def init_fun(_):
            return np.random.seed()

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)

    elif which_dataset == 'KNEE_RAW':
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4])
        # TODO: datasource changed to 01_101 since dataset01 is offline (H2 being down). Revert when dataset01 is up.
        raw_root = '/datasets01_101/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func)
        train_data = RawSliceData(
            raw_root + '/singlecoil_train', transform=data_transform, num_cols=368)
        valid_data = RawSliceData(
            raw_root + '/singlecoil_val', transform=data_transform, num_cols=368)

        def init_fun(_):
            return np.random.seed()

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)
        # raise NotImplementedError

    else:
        raise ValueError

    return train_loader, valid_loader


class SequentialSampler2(Sampler):
    r"""Samples elements sequentially, in the order of given list.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


def get_test_loader(batch_size, num_workers=2, pin_memory=False, which_dataset='KNEE'):
    if which_dataset in ('KNEE'):
        mask_func = FixedAccelerationMaskFunc([0.125], [4])
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform = DicomDataTransform(mask_func, None)
        test_data = Slice(
            data_transform,
            dicom_root,
            which='public_leaderboard',
            resolution=128,
            scan_type='all',
            num_volumes=None,
            num_rand_slices=None)

        def init_fun(_):
            return np.random.seed()

        data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)
    elif which_dataset == 'KNEE_RAW':
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4])
        # TODO: datasource changed to 01_101 since dataset01 is offline (H2 being down).
        #  Revert when dataset01 is up.
        raw_root = '/datasets01_101/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func)
        test_data = RawSliceData(
            raw_root + '/singlecoil_test', transform=data_transform, num_cols=368)

        def init_fun(_):
            return np.random.seed()

        data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=None,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=init_fun,
            pin_memory=pin_memory,
            drop_last=True)

    else:
        raise ValueError
    return data_loader
