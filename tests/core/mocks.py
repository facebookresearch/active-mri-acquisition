# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict

import numpy as np
import torch

import activemri.envs.envs as envs

cfg_json_str = """
{
    "data_location": "dummy_location",
    "reconstructor": {
        "cls": "tests.core.mocks.Reconstructor",
        "options": {
            "option1": 1,
            "option2": 0.5,
            "option3": "dummy",
            "option4": true
        },
        "checkpoint_fname": "null",
        "transform": "tests.core.mocks.transform"
    },
    "mask": {
        "function": "tests.core.mocks.mask_func",
        "args": {
            "size": 10,
            "how_many": 3
        }
    },
    "reward_metric": "ssim",
    "device": "cpu"
}
"""

config_dict = json.loads(cfg_json_str)


class Dataset:
    def __init__(self, tensor_size, length):
        self.tensor_size = tensor_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mock_kspace = (item + 1) * np.ones(
            (self.tensor_size, self.tensor_size, 2)  # 2 is for mocking (real, img.)
        )
        mock_mask = np.zeros(self.tensor_size)
        mock_ground_truth = mock_kspace + 1
        return mock_kspace, mock_mask, mock_ground_truth, {}, "fname", item


def make_data_init_fn(tensor_size, num_train, num_val, num_test):
    train_data = Dataset(tensor_size, num_train)
    val_data = Dataset(tensor_size, num_val)
    test_data = Dataset(tensor_size, num_test)

    def data_init_fn():
        return train_data, val_data, test_data

    return data_init_fn


# noinspection PyUnusedLocal
def mask_func(args, kspace_shapes, _rng, attrs=None):
    batch_size = len(kspace_shapes)
    mask = torch.zeros(batch_size, args["size"])
    mask[0, : args["how_many"]] = 1
    if batch_size > 1:
        mask[1, : args["how_many"] - 1] = 1
    return mask


def transform(kspace=None, mask=None, **_kwargs):
    if isinstance(mask, torch.Tensor):
        mask = mask.view(mask.shape[0], 1, -1, 1)
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(kspace, list):
        new_kspace = []
        for array in kspace:
            new_kspace.append(torch.from_numpy(array))
        return torch.stack(new_kspace), mask
    return kspace, mask


# noinspection PyMethodMayBeStatic
class Reconstructor:
    def __init__(self, **kwargs):
        self.option1 = kwargs["option1"]
        self.option2 = kwargs["option2"]
        self.option3 = kwargs["option3"]
        self.option4 = kwargs["option4"]
        self.weights = None
        self._eval = None
        self.device = None
        self.state_dict = {}

    def init_from_checkpoint(self, _checkpoint):
        self.weights = "init"

    def eval(self):
        self._eval = True

    def to(self, device):
        self.device = device

    def forward(self, kspace, mask):
        return {"reconstruction": kspace + mask}

    __call__ = forward

    def load_state_dict(self):
        pass


class MRIEnv(envs.ActiveMRIEnv):
    def __init__(
        self,
        num_parallel_episodes,
        loops_train,
        num_train=1,
        num_val=1,
        num_test=1,
        seed=None,
    ):
        super().__init__(
            (32, 64), num_parallel_episodes=num_parallel_episodes, seed=seed
        )
        self._num_loops_train_data = loops_train
        self._init_from_config_dict(config_dict)

        self._tensor_size = self._cfg["mask"]["args"]["size"]

        data_init_fn = make_data_init_fn(
            self._tensor_size, num_train, num_val, num_test
        )
        self._setup_data_handlers(data_init_fn)

    @staticmethod
    def _compute_score_given_tensors(
        reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        return {"ssim": (reconstruction - ground_truth).abs().sum().numpy()}
