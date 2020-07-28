import importlib
import json
import pathlib
import warnings

from typing import Any, Dict, List, Mapping, Tuple, Optional, Sized

import gym
import numpy as np
import torch
import torch.utils.data

import activemri.data.singlecoil_knee_data as scknee_data
import activemri.data.transforms
import activemri.envs.util


def update_masks_from_indices(masks: torch.Tensor, indices: np.ndarray):
    assert masks.shape[0] == indices.size
    for i, index in enumerate(indices):
        masks[i, :, index] = 1
    return masks


# noinspection PyUnusedLocal
class Reconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)

    def forward(
        self, zero_filled_image: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        return {"reconstruction": zero_filled_image, "return_vars": {"mask": mask}}

    def load_from_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        pass


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


# TODO Add option to resize default img size (need to pass this option to reconstructor)
# TODO Add option to control batch size (default 1)
# TODO See if there is a clean way to have access to current image indices from the env
class ActiveMRIEnv(gym.Env):
    def __init__(self, img_width, img_height):
        self._img_width = img_width
        self._img_height = img_height

        self._data_location = None
        self._reconstructor = None
        self._train_data_handler = None
        self._val_data_handler = None
        self._test_data_handler = None
        self._preprocess_func = None
        self._device = torch.device("cpu")

        self.horizon = None
        self.seed = None

        self.observation_space = None  # Observation will be a dictionary
        self.action_space = gym.spaces.Discrete(img_width)

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------
    def _init_from_config_dict(self, config: Mapping[str, Any]):
        self._data_location = config["data_location"]

        # Instantiating reconstructor
        reconstructor_cfg = config["reconstructor"]
        module = importlib.import_module(reconstructor_cfg["module"])
        reconstructor_cls = getattr(module, reconstructor_cfg["cls"])
        checkpoint_path = pathlib.Path(reconstructor_cfg["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path) if checkpoint_path.is_file() else None
        options = reconstructor_cfg["options"]
        if checkpoint and "options" in checkpoint:
            msg = (
                f"Checkpoint at {checkpoint_path.name} has an 'options' field. "
                f"This will override the options defined in configuration file."
            )
            warnings.warn(msg)
            options = checkpoint["options"]
            assert isinstance(options, dict)
        self._reconstructor = reconstructor_cls(**options)
        self._reconstructor.init_from_checkpoint(checkpoint)
        self._reconstructor.eval()
        self._reconstructor.to(self._device)

    def _init_from_config_file(self, config_filename: str):
        with open(config_filename, "rb") as f:
            self._init_from_config_dict(json.load(f))

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

    def valid_actions(self) -> List[int]:
        pass


class SingleCoilKneeRAWEnv(ActiveMRIEnv):
    def __init__(self):
        ActiveMRIEnv.__init__(self, 368, 640)
        # self._init_from_config_file("configs/single-coil-knee-raw.json")
        self._init_from_config_file("configs/miccai_raw.json")
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
