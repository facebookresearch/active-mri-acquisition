# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib

import h5py
import numpy as np
import torch
import torch.utils.data


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


class RawSliceData(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transform,
        num_cols=None,
        num_volumes=None,
        num_rand_slices=None,
        custom_split=None,
    ):
        self.transform = transform
        self.examples = []

        self.num_rand_slices = num_rand_slices
        self.rng = np.random.RandomState(1234)

        files = []
        for fname in list(pathlib.Path(root).iterdir()):
            data = h5py.File(fname, "r")
            if num_cols is not None and data["kspace"].shape[2] != num_cols:
                continue
            files.append(fname)

        if custom_split is not None:
            split_info = []
            with open(f"data/splits/raw_{custom_split}.txt") as f:
                for line in f:
                    split_info.append(line.rsplit("\n")[0])
            files = [f for f in files if f.name in split_info]

        if num_volumes is not None:
            self.rng.shuffle(files)
            files = files[:num_volumes]

        for volume_i, fname in enumerate(sorted(files)):
            data = h5py.File(fname, "r")
            kspace = data["kspace"]

            if num_rand_slices is None:
                num_slices = kspace.shape[0]
                self.examples += [(fname, slice) for slice in range(num_slices)]
            else:
                slice_ids = list(range(kspace.shape[0]))
                self.rng.seed(seed=volume_i)
                self.rng.shuffle(slice_ids)
                self.examples += [
                    (fname, slice) for slice in slice_ids[:num_rand_slices]
                ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice]
            return self.transform(kspace, data.attrs)


class RawDataTransform:
    def __init__(self, mask_func, fixed_seed=None, seed_per_image=False):
        self.mask_func = mask_func
        self.fixed_seed = fixed_seed
        self.seed_per_image = seed_per_image

    def __call__(self, kspace, attrs):
        kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
        kspace = ifftshift(kspace, dim=(0, 1))
        image = torch.ifft(kspace, 2, normalized=False)
        image = ifftshift(image, dim=(0, 1))
        # norm = torch.sqrt(image[..., 0] ** 2 + image[..., 1] ** 2).max()
        # 5.637766165023095e-08, 7.072103529760345e-07, 5.471710210258607e-06
        # normalize by the mean norm of training images.
        image /= 7.072103529760345e-07
        kspace /= 7.072103529760345e-07
        shape = np.array(kspace.shape)
        seed = (
            int(1009 * image.sum().abs())
            if self.fixed_seed is None and self.seed_per_image
            else self.fixed_seed
        )
        mask = self.mask_func(shape, seed)
        return mask, image, kspace
