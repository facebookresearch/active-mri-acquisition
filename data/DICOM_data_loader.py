import json
import os
import pickle

import numpy as np
import torch

from torch.utils.data import Dataset

import models.fft_utils


class Slice(Dataset):

    def __init__(self,
                 transform,
                 dicom_root,
                 which='train',
                 resolution=320,
                 scan_type=None,
                 num_volumes=None,
                 num_rand_slices=None):
        self.transform = transform
        self.dataset = _DicomDataset(
            dicom_root / str(resolution) / which, scan_type, num_volumes=num_volumes)
        self.num_slices = self.dataset.metadata['num_slices']
        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState()

    def __getitem__(self, i):
        i = int(i)

        if self.num_rand_slices is None:
            volume_i, slice_i = divmod(i, self.num_slices)
        else:
            volume_i = (i * self.num_slices // self.num_rand_slices) // self.num_slices
            slice_ids = list(range(self.num_slices))
            self.rng.seed(seed=volume_i)
            self.rng.shuffle(slice_ids)
            slice_i = slice_ids[i % self.num_rand_slices]

        volume, volume_metadata = self.dataset[volume_i]
        slice = volume[slice_i:slice_i + 1]
        slice = slice.astype(np.float32)
        return self.transform(slice, volume_metadata['mean'], volume_metadata['std'])

    def __len__(self):
        if self.num_rand_slices is None:
            return len(self.dataset) * self.num_slices
        else:
            return len(self.dataset) * self.num_rand_slices


class SliceWithPrecomputedMasks(Dataset):
    """ This refers to a dataset of pre-computed masks where the Slice object had the following
        characteristics:
            -resolution: 128, scan_type='all', num_volumes=None, num_rand_slices=None

        Only a subset of the masks where computed, specified by the list `self.available_masks`.
    """

    def __init__(self, transform, dicom_root, masks_dir, which='train'):
        self.transform = transform
        self.dataset = _DicomDataset(dicom_root / str(128) / which, 'all', num_volumes=None)
        self.num_slices = self.dataset.metadata['num_slices']
        self.rng = np.random.RandomState()

        self.masks_location = masks_dir
        self.masks_location = os.path.join(self.masks_location, which)
        # File format is masks_{begin_idx}-{end_idx}.p. The lines below obtain all begin_idx
        self.maskfile_begin_indices = sorted([
            int(x.split('_')[1].split('-')[0])
            for x in os.listdir(self.masks_location)
            if 'masks' in x
        ])
        self.maskfile_end_indices = sorted([
            int(x.split('_')[1].split('-')[1].split('.')[0])
            for x in os.listdir(self.masks_location)
            if 'masks' in x
        ])
        self.masks_per_file = self.maskfile_begin_indices[1] - self.maskfile_begin_indices[0]

    def __getitem__(self, i):
        available_mask_index = i // self.masks_per_file
        mask_file_index = self.maskfile_begin_indices[available_mask_index]
        mask_index = i % self.masks_per_file

        volume_i, slice_i = divmod(i, self.num_slices)
        volume, volume_metadata = self.dataset[volume_i]
        slice = volume[slice_i:slice_i + 1]
        slice = slice.astype(np.float32)
        _, image = self.transform(slice, volume_metadata['mean'], volume_metadata['std'])

        # Now load the pre-computed mask
        filename = os.path.join(
            self.masks_location,
            f'masks_{mask_file_index}-{self.maskfile_end_indices[available_mask_index]}.p')
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            mask = torch.from_numpy(data[mask_index]).view(1, 1, 128).float()

        return mask, image

    def __len__(self):
        return self.maskfile_end_indices[-1]


class DicomDataTransform:

    # If `seed` is none and `seed_per_image` is True, masks will be generated with a unique seed
    # per image, computed as `seed = int( 1009 * image.sum().abs())`.
    def __init__(self, mask_func, fixed_seed=None, seed_per_image=False):
        self.mask_func = mask_func
        self.fixed_seed = fixed_seed
        self.seed_per_image = seed_per_image

    def __call__(self, image, mean, std):
        image = (image - mean) / (std + 1e-12)
        image = torch.from_numpy(image)
        image = models.fft_utils.clamp(image)
        shape = np.array(image.shape)
        seed = int(1009 * image.sum().abs()) if self.fixed_seed is None and self.seed_per_image \
            else self.fixed_seed
        mask = self.mask_func(shape, seed) if self.mask_func is not None else None
        return mask, image


class _DicomDataset:

    def __init__(self, root, scan_type=None, num_volumes=None):
        self.metadata = json.load(open(os.path.join(root, 'metadata.json')))
        shape = len(self.metadata['volumes']), self.metadata['num_slices'], self.metadata[
            'resolution'], self.metadata['resolution']
        self.volumes = np.memmap(os.path.join(root, 'data.bin'), self.metadata['dtype'],
                                 'r').reshape(shape)

        volume_ids = []
        for i, volume in enumerate(self.metadata['volumes']):
            if scan_type == 'all' or volume['scan_type'] == scan_type:
                volume_ids.append(i)

        if num_volumes is not None:
            rng = np.random.RandomState(1234)
            rng.shuffle(volume_ids)
            volume_ids = volume_ids[:num_volumes]

        self.volume_ids = {i: id for i, id in enumerate(volume_ids)}

    def __getitem__(self, i):
        """ returns (data: 4d array, metadata: dict) """
        id = self.volume_ids[i]
        return self.volumes[id], self.metadata['volumes'][id]

    def __len__(self):
        return len(self.volume_ids)
