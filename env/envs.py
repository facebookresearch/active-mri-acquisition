import importlib
import json

from typing import Dict, Tuple

import gym
import numpy as np
import torch


class ActiveMRIEnv(gym.Env):
    def __init__(self):
        self._config_file = None
        self._dataset_location = None
        self._reconstructor = None
        self._device = None
        pass

    def reset(self,) -> Dict[str, np.ndarray]:
        pass

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, bool]:
        pass

    def render(self, mode="human"):
        pass

    def _read_config_file(self):
        with open(self._config_file, "rb") as f:
            config = json.load(f)
        self._dataset_location = config["dataset_location"]
        module = importlib.import_module(config["reconstructor_module"])
        reconstructor_cls = getattr(module, config["reconstructor_cls"])
        self._reconstructor = reconstructor_cls(**config["reconstructor_options"])
        self._device = torch.cuda.device(config["device"])


class SingleCoilKneeRAWEnv(ActiveMRIEnv):
    def __init__(self):
        ActiveMRIEnv.__init__(self)
        self._config_file = "configs/single-coil-knee-raw.json"
        self._read_config_file()

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
