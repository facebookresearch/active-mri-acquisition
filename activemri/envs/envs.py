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


class CyclicSampler(torch.utils.data.Sampler):
    def __init__(
        self, data_source: Sized, order: Optional[Sized] = None, loops: int = 1,
    ):
        torch.utils.data.Sampler.__init__(self, data_source)
        assert loops > 0
        assert order is None or len(order) == len(data_source)
        self.data_source = data_source
        self.order = order if order is not None else range(len(self.data_source))
        self.loops = loops

    def __iterator(self):
        for _ in range(self.loops):
            for j in self.order:
                yield j

    def __iter__(self):
        return iter(self.__iterator())

    def __len__(self):
        return len(self.data_source) * self.loops


class DataHandler:
    def __init__(self, data_source, seed, batch_size=1, loops=1):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(data_source))
        sampler = CyclicSampler(data_source, order, loops=loops)
        self.__data_loader = torch.utils.data.DataLoader(
            data_source, batch_size=batch_size, sampler=sampler
        )
        self.__iter = iter(self.__data_loader)

    def reset(self):
        self.__iter = iter(self.__data_loader)

    def __iter__(self):
        return self.__iter

    def __next__(self):
        return self.__data_loader


# TODO Add option to resize default img size
# TODO Add option to control batch size (default 1)
# TODO See if there is a clean way to have access to current image indices from the env
class ActiveMRIEnv(gym.Env):
    def __init__(self):
        self._data_location = None
        self._reconstructor = None
        self._train_data_handler = None
        self._val_data_handler = None
        self._test_data_handler = None
        self._device = torch.device("cpu")
        self._img_width = None
        self._img_height = None

        self.horizon = None
        self.seed = None

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _init_from_dict(self, config: Dict[str, Any]):
        self._data_location = config["data_location"]

        # Instantiating reconstructor
        reconstructor_config = config["reconstructor"]
        module = importlib.import_module(reconstructor_config["module"])
        reconstructor_cls = getattr(module, reconstructor_config["cls"])
        self._reconstructor = reconstructor_cls(**reconstructor_config["options"])

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
        self.seed = seed

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

    def restart_train(self):
        self._train_data_handler.reset()

    def restart_val(self):
        self._val_data_handler.reset()

    def restart_test(self):
        self._test_data_handler.reset()


class SingleCoilKneeRAWEnv(ActiveMRIEnv):
    def __init__(self):
        ActiveMRIEnv.__init__(self)
        self._init_from_config_file("configs/single-coil-knee-raw.json")
        self._img_width = 368
        self._img_height = 640
        self.__setup_data_handlers()

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------
    def __setup_data_handlers(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path.joinpath("knee_singlecoil_train")
        val_and_test_path = root_path.joinpath("knee_singlecoil_val")
        transform = activemri.data.transforms.raw_transform_miccai20
        # Setting up training data loader
        train_data = scknee_data.RawSliceData(train_path, transform)
        self._train_data_handler = DataHandler(
            train_data, self.seed, batch_size=1, loops=1000
        )

        # Setting up val data loader
        val_data = scknee_data.RawSliceData(
            val_and_test_path, transform, custom_split="val"
        )
        self._val_data_handler = DataHandler(
            val_data, self.seed + 1 if self.seed else None, batch_size=1, loops=1
        )
        # Setting up test data loader
        test_data = scknee_data.RawSliceData(
            val_and_test_path, transform, custom_split="test"
        )
        self._test_data_handler = DataHandler(
            test_data, self.seed + 2 if self.seed else None, batch_size=1, loops=1
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
