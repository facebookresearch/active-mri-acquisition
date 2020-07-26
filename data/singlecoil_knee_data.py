import os
import pathlib

import h5py
import json
import numpy as np
import torch.utils.data

from typing import Callable, Optional


class RawSliceData(torch.utils.data.Dataset):
    def __init__(
        self,
        root: pathlib.Path,
        transform: Callable,
        num_cols: Optional[int] = None,
        num_volumes: Optional[int] = None,
        num_rand_slices: Optional[int] = None,
        custom_split: Optional[str] = None,
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
            with open(f"data/splits/knee_singlecoil/{custom_split}.txt") as f:
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


class Slice(torch.utils.data.Dataset):
    def __init__(
        self,
        transform: Callable,
        root: str,
        split: str = "train",
        resolution: int = 128,
        scan_type: Optional[str] = None,
        num_volumes: Optional[int] = None,
        num_rand_slices: Optional[int] = None,
    ):
        self.transform = transform
        self.dataset = _DicomDataset(
            root / str(resolution) / split, scan_type, num_volumes=num_volumes
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


class _DicomDataset:
    def __init__(
        self,
        root: str,
        scan_type: Optional[str] = None,
        num_volumes: Optional[int] = None,
    ):
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
