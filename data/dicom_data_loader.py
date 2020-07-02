import json
import os

import numpy as np
import torch

from torch.utils.data import Dataset

import models.fft_utils


class Slice(Dataset):
    def __init__(
        self,
        transform,
        dicom_root,
        which="train",
        resolution=320,
        scan_type=None,
        num_volumes=None,
        num_rand_slices=None,
    ):
        self.transform = transform
        self.dataset = _DicomDataset(
            dicom_root / str(resolution) / which, scan_type, num_volumes=num_volumes
        )
        self.num_slices = self.dataset.metadata["num_slices"]
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
        slice = volume[slice_i : slice_i + 1]
        slice = slice.astype(np.float32)
        return self.transform(slice, volume_metadata["mean"], volume_metadata["std"])

    def __len__(self):
        if self.num_rand_slices is None:
            return len(self.dataset) * self.num_slices
        else:
            return len(self.dataset) * self.num_rand_slices


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
        image = models.fft_utils.dicom_to_0_1_range(image)
        shape = np.array(image.shape)
        seed = (
            int(1009 * image.sum().abs())
            if self.fixed_seed is None and self.seed_per_image
            else self.fixed_seed
        )
        mask = self.mask_func(shape, seed) if self.mask_func is not None else None
        image = torch.cat([image, torch.zeros_like(image)], dim=0)
        return mask, image


class _DicomDataset:
    def __init__(self, root, scan_type=None, num_volumes=None):
        self.metadata = json.load(open(os.path.join(root, "metadata.json")))
        shape = (
            len(self.metadata["volumes"]),
            self.metadata["num_slices"],
            self.metadata["resolution"],
            self.metadata["resolution"],
        )
        self.volumes = np.memmap(
            os.path.join(root, "data.bin"), self.metadata["dtype"], "r"
        ).reshape(shape)

        volume_ids = []
        for i, volume in enumerate(self.metadata["volumes"]):
            if scan_type == "all" or volume["scan_type"] == scan_type:
                volume_ids.append(i)

        if num_volumes is not None:
            rng = np.random.RandomState(1234)
            rng.shuffle(volume_ids)
            volume_ids = volume_ids[:num_volumes]

        self.volume_ids = {i: id for i, id in enumerate(volume_ids)}

    def __getitem__(self, i):
        """ returns (data: 4d array, metadata: dict) """
        id = self.volume_ids[i]
        return self.volumes[id], self.metadata["volumes"][id]

    def __len__(self):
        return len(self.volume_ids)
