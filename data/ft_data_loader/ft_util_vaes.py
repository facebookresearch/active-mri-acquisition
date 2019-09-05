import pathlib
import pickle

import h5py
import numpy as np
import torch
import json
import os

from torch.utils.data import Dataset, RandomSampler
"""
def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, acc, sample_n=2, centred=False):

    # Code from: https://github.com/js3611/Deep-MRI-Reconstruction
    # Sampling density estimated from implementation of kt FOCUSS
    # shape: tuple - of form (..., nx, ny)
    # acc: float - doesn't have to be integer 4, 8, etc..

    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask

def ifft2(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def rfft2(data):
    zeros = torch.zeros_like(data)
    data = torch.stack([data, zeros], dim=3)
    return fft2(data)

def fft2(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data
"""


def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def roll(x, shift, dim):
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


class FixedOrderRandomSampler(RandomSampler):

    def __init__(self, data_source):
        super().__init__(data_source)
        n = len(self.data_source)
        self.rand_order = torch.randperm(n).tolist()

    def __iter__(self):
        return iter(self.rand_order)


class MaskFunc:

    def __init__(self, center_fractions, accelerations, which_dataset, random_num_lines=False):
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.random_num_lines = random_num_lines

        # The lines below give approx. 4x acceleration on average.
        self.min_lowf_lines = 6 if which_dataset != 'KNEE_RAW' else 16
        self.max_lowf_lines = 16 if which_dataset != 'KNEE_RAW' else 44
        self.highf_beta_alpha = 1
        self.highf_beta_beta = 5

        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        # Determine number of low and high frequency lines to scan
        if self.random_num_lines:
            # These are guaranteed to be an even number (useful for symmetric masks)
            num_low_freqs = self.rng.choice(range(self.min_lowf_lines, self.max_lowf_lines, 2))
            num_high_freqs = int(
                self.rng.beta(self.highf_beta_alpha, self.highf_beta_beta) *
                (num_cols - num_low_freqs) // 2) * 2
        else:
            choice = self.rng.randint(0, len(self.accelerations))
            center_fraction = self.center_fractions[choice]
            acceleration = self.accelerations[choice]

            num_low_freqs = int(round(num_cols * center_fraction))
            num_high_freqs = int(num_cols // acceleration - num_low_freqs)

        # Create the mask
        mask = self.create_lf_focused_mask(num_cols, num_high_freqs, num_low_freqs)

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-1] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        p = num_high_freqs / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < p
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        return mask


class BasicMaskFunc(MaskFunc):

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        hf_cols = self.rng.choice(
            np.arange(num_cols - num_low_freqs), num_high_freqs, replace=False)
        hf_cols[hf_cols >= (num_cols - num_low_freqs + 1) // 2] += num_low_freqs
        mask[hf_cols] = True
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        mask = np.fft.ifftshift(mask, axes=0)
        return mask


class SymmetricUniformChoiceMaskFunc(MaskFunc):

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        num_cols //= 2
        num_low_freqs //= 2
        num_high_freqs //= 2
        hf_cols = self.rng.choice(
            np.arange(num_cols - num_low_freqs), num_high_freqs, replace=False)
        mask[hf_cols] = True
        pad = (num_cols - num_low_freqs)
        mask[pad:num_cols] = True
        mask[:-(num_cols + 1):-1] = mask[:num_cols]
        mask = np.fft.ifftshift(mask, axes=0)
        return mask


class UniformGridMaskFunc(MaskFunc):

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        acceleration = self.rng.choice([4, 8, 16])
        hf_cols = np.arange(acceleration, num_cols, acceleration)
        mask[hf_cols] = True
        mask[:num_low_freqs // 2] = mask[-(num_low_freqs // 2):] = True
        return mask


class SymmetricUniformGridMaskFunc(MaskFunc):

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        acceleration = self.rng.choice([4, 8, 16])
        num_cols //= 2
        num_low_freqs //= 2
        hf_cols = np.arange(acceleration, num_cols, acceleration)
        mask[hf_cols] = True
        mask[:num_low_freqs] = True
        mask[:-(num_cols + 1):-1] = mask[:num_cols]
        return mask


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
        self.available_masks = sorted([
            int(x.split('_')[1].split('-')[0])
            for x in os.listdir(self.masks_location)
            if 'masks' in x
        ])
        self.masks_per_file = self.available_masks[1] - self.available_masks[0]

    def __getitem__(self, i):
        available_mask_index = i // self.masks_per_file
        mask_file_index = self.available_masks[available_mask_index]
        mask_index = i % self.masks_per_file

        volume_i, slice_i = divmod(i, self.num_slices)
        volume, volume_metadata = self.dataset[volume_i]
        slice = volume[slice_i:slice_i + 1]
        slice = slice.astype(np.float32)
        _, image = self.transform(slice, volume_metadata['mean'], volume_metadata['std'])

        # Now load the pre-computed mask
        filename = os.path.join(
            self.masks_location,
            f'masks_{mask_file_index}-{mask_file_index + self.masks_per_file}.p')
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            mask = torch.from_numpy(data[mask_index]).view(1, 1, 128).float()

        return mask, image

    def __len__(self):
        return self.masks_per_file * len(self.available_masks)


class RawSliceData(Dataset):

    def __init__(self, root, transform, num_cols=None, num_volumes=None, num_rand_slices=None):
        self.transform = transform
        self.examples = []

        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, 'r')
            if num_cols is not None and data['kspace'].shape[2] != num_cols:
                continue
            files.append(fname)

        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, 'r')
            kspace = data['kspace']

            if num_rand_slices is None:
                num_slices = kspace.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            else:
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [(fname, slice) for slice in slice_ids[:num_rand_slices]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            return self.transform(kspace, data.attrs)


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
        shape = np.array(image.shape)
        seed = int(1009 * image.sum().abs()) if self.fixed_seed is None and self.seed_per_image \
            else self.fixed_seed
        mask = self.mask_func(shape, seed) if self.mask_func is not None else None
        return mask, image

    def postprocess(self, data, hps):
        raise NotImplementedError('Need to fix sign leakage here')
        # inputs = data[1]
        # mask = data[0].repeat(1, 1, hps.resolution, 1)
        # inputs = (inputs.clamp_(-2, 4) + 2) / 6
        # inputs = torch.cat((inputs, torch.zeros(inputs.shape)), 1)
        # inputs = inputs.cuda()
        # mask = mask.cuda()
        # mask_k_space = mask
        # mask = mask.unsqueeze(4).repeat(1, 1, 1, 1, 2)
        # kspace = torch.fft(inputs.permute(0, 2, 3, 1), 2, normalized=False).unsqueeze(1)
        # masked_kspace = kspace * mask
        # masked_image = torch.ifft(masked_kspace, 2, normalized=False).squeeze(1).permute(0, 3, 1, 2)
        # return inputs, masked_image, mask_k_space


class RawDataTransform:

    def __init__(self, mask_func, seed=None):
        self.mask_func = mask_func
        self.seed = seed

    def __call__(self, kspace, attrs):
        kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
        kspace = ifftshift(kspace, dim=(0, 1))
        image = torch.ifft(kspace, 2, normalized=False)
        image = ifftshift(image, dim=(0, 1))
        norm = torch.sqrt(image[..., 0]**2 + image[..., 1]**2).max()
        image /= norm
        kspace /= norm
        shape = np.array(kspace.shape)
        mask = self.mask_func(shape, self.seed)
        return mask, image, kspace

    def postprocess(self, data, hps):
        mask, inputs, kspace = data
        mask = data[0].repeat(1, 1, kspace.shape[1], 1)
        inputs = inputs.cuda().permute(0, 3, 1, 2)
        kspace = kspace.cuda().unsqueeze(1)
        mask = mask.cuda()
        mask_k_space = mask
        mask = mask.unsqueeze(4).repeat(1, 1, 1, 1, 2)
        masked_kspace = kspace * mask
        masked_image = ifftshift(torch.ifft(masked_kspace, 2, normalized=False), dim=(2, 3))
        masked_image = masked_image.squeeze(1).permute(0, 3, 1, 2)
        return inputs, masked_image, mask_k_space


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
