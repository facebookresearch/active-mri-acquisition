import importlib
import json

from typing import Any, Dict, Tuple, Optional

import gym
import numpy as np
import torch


# TODO Add option to resize default img size
class ActiveMRIEnv(gym.Env):
    def __init__(self):
        self._dataset_location = None
        self._reconstructor = None
        self._train_data_loader = None
        self._valid_data_loader = None
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
    # Private methods
    # -------------------------------------------------------------------------
    def _init_from_dict(self, config: Dict[str, Any]):
        self._dataset_location = config["dataset_location"]

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
