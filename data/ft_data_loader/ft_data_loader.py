"""
Source : https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
"""
import logging
import pathlib
import os

import torch
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from .ft_util_vaes import MaskFunc, DicomDataTransform, Slice, FixedAccelerationMaskFunc, \
    SymmetricUniformChoiceMaskFunc, UniformGridMaskFunc, SymmetricUniformGridMaskFunc, \
    RawDataTransform, RawSliceData, SliceWithPrecomputedMasks


def get_mask_func(mask_type):
    # Whether the number of lines is random or not
    random_num_lines = (mask_type[-4:] == '_rnl')
    if 'fixed_acc' in mask_type:
        # First two parameters are ignored if `random_num_lines` is True
        logging.info(f'Mask is fixed acceleration mask with random_num_lines={random_num_lines}.')
        return FixedAccelerationMaskFunc([0.125], [4], random_num_lines=random_num_lines)
    if 'symmetric_choice' in mask_type:
        logging.info(f'Mask is symmetric uniform choice with random_num_lines={random_num_lines}.')
        return SymmetricUniformChoiceMaskFunc([0.125], [4], random_num_lines=random_num_lines)
    if 'symmetric_grid' in mask_type:
        logging.info(f'Mask is symmetric grid.')
        return SymmetricUniformGridMaskFunc([], [], random_num_lines=True)
    if 'grid' in mask_type:
        logging.info(f'Mask is grid (not symmetric).')
        return UniformGridMaskFunc([], [], random_num_lines=True)
    raise ValueError(f'Invalid mask type: {mask_type}.')


def get_train_valid_loader(batch_size,
                           num_workers=4,
                           pin_memory=False,
                           which_dataset='KNEE',
                           mask_type='fixed_acc'):

    if which_dataset == 'KNEE_PRECOMPUTED_MASKS':
        dicom_root = pathlib.Path('/checkpoint/jzb/data/mmap')
        data_transform_train = DicomDataTransform(None, fixed_seed=None, seed_per_image=True)
        data_transform_valid = DicomDataTransform(
            FixedAccelerationMaskFunc([0.125], [4]), fixed_seed=None, seed_per_image=True)
        train_data = SliceWithPrecomputedMasks(data_transform_train, dicom_root, which='train')
        valid_data = Slice(
            data_transform_valid,
            dicom_root,
            which='val',
            resolution=128,
            scan_type='all',
            num_volumes=None,
            num_rand_slices=None)

    elif which_dataset == 'KNEE':
        mask_func = get_mask_func(mask_type)
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
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4])
        raw_root = '/datasets01_101/fastMRI/112718'
        if not os.path.isdir(raw_root):
            raise ImportError(raw_root + ' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func)
        train_data = RawSliceData(
            raw_root + '/singlecoil_train', transform=data_transform, num_cols=368)
        valid_data = RawSliceData(
            raw_root + '/singlecoil_val', transform=data_transform, num_cols=368)
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


def get_test_loader(batch_size,
                    num_workers=2,
                    pin_memory=False,
                    which_dataset='KNEE',
                    mask_type='fixed_acc'):
    if which_dataset in ('KNEE'):
        mask_func = get_mask_func(mask_type)
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
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4])
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
