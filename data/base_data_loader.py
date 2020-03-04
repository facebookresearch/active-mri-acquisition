import pathlib
import os

import torch
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler

from .dicom_data_loader import DicomDataTransform, Slice
from .raw_data_loader import RawDataTransform, RawSliceData
from .masking_utils import get_mask_func


def get_train_valid_loader(batch_size,
                           num_workers=4,
                           pin_memory=False,
                           which_dataset='KNEE',
                           mask_type='basic',
                           rnl_params=None,
                           num_volumes_train=None,
                           num_volumes_val=None):

    if which_dataset == 'KNEE':
        mask_func = get_mask_func(mask_type, which_dataset, rnl_params=rnl_params)
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform = DicomDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
        train_data = Slice(
            data_transform,
            dicom_root,
            which='train',
            resolution=128,
            scan_type='all',
            num_volumes=num_volumes_train,
            num_rand_slices=None)
        valid_data = Slice(
            data_transform,
            dicom_root,
            which='val',
            resolution=128,
            scan_type='all',
            num_volumes=num_volumes_val,
            num_rand_slices=None)

    elif which_dataset == 'KNEE_RAW':
        mask_func = get_mask_func(mask_type, which_dataset, rnl_params=rnl_params)
        raw_root = '/datasets01/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func, fixed_seed=None, seed_per_image=False)
        train_data = RawSliceData(
            raw_root + '/singlecoil_train',
            transform=data_transform,
            num_cols=368,
            num_volumes=num_volumes_train)
        data_transform = RawDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
        valid_data = RawSliceData(
            raw_root + '/singlecoil_val',
            transform=data_transform,
            num_cols=368,
            num_volumes=num_volumes_val,
            custom_split='val')
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
                    mask_type='basic',
                    rnl_params=None):
    if which_dataset in ('KNEE'):
        mask_func = get_mask_func(mask_type, which_dataset, rnl_params=rnl_params)
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
        mask_func = get_mask_func(mask_type, which_dataset, rnl_params=rnl_params)
        raw_root = '/datasets01/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func, fixed_seed=None, seed_per_image=True)
        test_data = RawSliceData(
            raw_root + '/singlecoil_val',
            transform=data_transform,
            num_cols=368,
            custom_split='test')

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
