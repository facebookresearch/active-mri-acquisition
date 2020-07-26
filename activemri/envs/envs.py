import importlib
import json
import pathlib

from typing import Any, Dict, Tuple, Optional, Sized

import gym
import numpy as np
import torch
import torch.utils.data

import activemri.data.singlecoil_knee_data as scknee_data
import activemri.data.transforms


# noinspection PyUnusedLocal
class NullReconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)

    def forward(self, zero_filled_image, mask, **kwargs):
        return {"reconstruction": zero_filled_image, "return_vars": {"mask": mask}}


def update_masks_from_indices(masks: torch.Tensor, indices: np.ndarray):
    assert masks.shape[0] == indices.size
    for i, index in enumerate(indices):
        masks[i, :, index] = 1
    return masks


def _infinite_generator(a_list):
    while True:
        for j in a_list:
            yield j


class CyclicSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Sized,
        rng: Optional[np.random.RandomState] = None,
        loops: int = 1000,
    ):
        torch.utils.data.Sampler.__init__(self, data_source)
        assert loops > 0
        self.data_source = data_source
        self.order = (
            rng.permutation(len(self.data_source))
            if rng
            else range(len(self.data_source))
        )
        self.loops = loops

    def __iter__(self):
        return iter(_infinite_generator(self.order))

    def __len__(self):
        return len(self.data_source) * self.loops


# TODO Add option to resize default img size
# TODO Add option to control batch size (default 1)
# TODO See if there is a clean way to have access to current image indices from the env
# TODO Add code to restart data loaders
class ActiveMRIEnv(gym.Env):
    def __init__(self):
        self._data_location = None
        self._reconstructor = None
        self._train_data_loader = None
        self._val_data_loader = None
        self._test_data_loader = None
        self._device = torch.device("cpu")
        self._img_width = None
        self._img_height = None
        self._idx_img_train = 0
        self._idx_img_valid = 0
        self._idx_img_test = 0

        self.horizon = None
        self.rng = np.random.RandomState()

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _init_from_dict(self, config: Dict[str, Any]):
        self._data_location = config["data_location"]

        # Instantiating reconstructor
        module = importlib.import_module(config["reconstructor_module"])
        reconstructor_cls = getattr(module, config["reconstructor_cls"])
        self._reconstructor = reconstructor_cls(**config["reconstructor_options"])

    def _init_from_config_file(self, config_filename: str):
        with open(config_filename, "rb") as f:
            self._init_from_dict(json.load(f))

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------
    def reset(self,) -> Dict[str, np.ndarray]:
        pass

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def render(self, mode="human"):
        pass

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def reset_val(self,) -> Dict[str, np.ndarray]:
        pass

    def step_val(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def reset_test(self,) -> Dict[str, np.ndarray]:
        pass

    def step_test(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def restart_train_data_loader(self):
        self._idx_img_train = 0

    def restart_valid_data_loader(self):
        self._idx_img_valid = 0

    def restart_test_data_loader(self):
        self._idx_img_test = 0


class SingleCoilKneeRAWEnv(ActiveMRIEnv):
    def __init__(self):
        ActiveMRIEnv.__init__(self)
        self._init_from_config_file("configs/single-coil-knee-raw.json")
        self._img_width = 368
        self._img_height = 640
        self.__setup_dataloaders()

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------
    def __setup_dataloaders(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path.joinpath("knee_singlecoil_train")
        val_and_test_path = root_path.joinpath("knee_singlecoil_val")
        transform = activemri.data.transforms.raw_transform_miccai20
        # Setting up training data loader
        train_data = scknee_data.RawSliceData(train_path, transform)
        train_sampler = CyclicSampler(train_data, self.rng)
        self._train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, sampler=train_sampler
        )
        # Setting up val data loader
        val_data = scknee_data.RawSliceData(
            val_and_test_path, transform, custom_split="val"
        )
        val_sampler = CyclicSampler(val_data, self.rng, loops=1)
        self._val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=1, sampler=val_sampler
        )
        # Setting up test data loader
        test_data = scknee_data.RawSliceData(
            val_and_test_path, transform, custom_split="test"
        )
        test_sampler = CyclicSampler(test_data, self.rng, loops=1)
        self._test_data_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, sampler=test_sampler
        )

    def reset(self) -> Dict[str, np.ndarray]:
        pass

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    import sys

    sys.path.append("/private/home/lep/code/Active_Acquisition")
    foo = SingleCoilKneeRAWEnv()
    print("boo")
