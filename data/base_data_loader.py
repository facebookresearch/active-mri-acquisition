"""
Source : https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
"""
import logging
import pathlib
import os

import torch
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from .DICOM_data_loader import DicomDataTransform, Slice, SliceWithPrecomputedMasks
from .RAW_data_loader import RawDataTransform, RawSliceData
from .masking_utils import get_mask_func

class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None


def get_train_valid_loader(batch_size,
                           num_workers=4,
                           pin_memory=False,
                           which_dataset='KNEE',
                           mask_type='basic',
                           masks_dir=None):

    if which_dataset == 'KNEE_PRECOMPUTED_MASKS':
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform_train = DicomDataTransform(None, fixed_seed=None, seed_per_image=True)
        data_transform_valid = DicomDataTransform(None, fixed_seed=None, seed_per_image=True)
        train_data = SliceWithPrecomputedMasks(
            data_transform_train, dicom_root, masks_dir, which='train')
        valid_data = SliceWithPrecomputedMasks(
            data_transform_valid, dicom_root, masks_dir, which='val')

    elif which_dataset == 'KNEE':
        mask_func = get_mask_func(mask_type, which_dataset)
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform = DicomDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
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

    elif which_dataset == 'KNEE_RAW':
        mask_func = get_mask_func(mask_type, which_dataset)
        raw_root = '/datasets01_101/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
        train_data = RawSliceData(
            raw_root + '/singlecoil_train',
            transform=data_transform,
            num_cols=368,
            num_volumes=None)
        valid_data = RawSliceData(
            raw_root + '/singlecoil_val', transform=data_transform, num_cols=368, num_volumes=None)
    else:
        raise ValueError

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

    return train_loader, valid_loader


def get_test_loader(batch_size,
                    num_workers=2,
                    pin_memory=False,
                    which_dataset='KNEE',
                    mask_type='basic'):
    if which_dataset in ('KNEE'):
        mask_func = get_mask_func(mask_type, which_dataset)
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform = DicomDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
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
        mask_func = get_mask_func(mask_type, which_dataset)
        raw_root = '/datasets01_101/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
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
