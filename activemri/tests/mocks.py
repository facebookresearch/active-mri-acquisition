import json
from typing import List, Dict

import numpy as np
import torch

import activemri.envs.envs as envs

cfg_json_str = """
{
    "data_location": "dummy_location",
    "reconstructor": {
        "cls": "activemri.tests.mocks.Reconstructor",
        "options": {
            "option1": 1,
            "option2": 0.5,
            "option3": "dummy",
            "option4": true
        },
        "checkpoint_path": "null",
        "transform": "activemri.tests.mocks.transform"
    },
    "mask": {
        "function": "activemri.tests.mocks.mask_func",
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


# noinspection PyUnresolvedReferences
class Dataset:
    def __init__(self, tensor_size, length):
        self.tensor_size = tensor_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mock_kspace = (item + 1) * torch.ones(
            self.tensor_size, self.tensor_size, 2  # 2 is for mocking (real, img.)
        )
        mock_mask = torch.zeros(self.tensor_size)
        mock_ground_truth = mock_kspace + 1
        return mock_kspace, mock_mask, mock_ground_truth, {}, "fname", item


def make_data_init_fn(tensor_size, num_train, num_val, num_test):
    train_data = Dataset(tensor_size, num_train)
    val_data = Dataset(tensor_size, num_val)
    test_data = Dataset(tensor_size, num_test)

    def data_init_fn():
        return train_data, val_data, test_data

    return data_init_fn


# noinspection PyUnresolvedReferences
def mask_func(args, batch_size, _rng):
    mask = torch.zeros(batch_size, args["size"])
    mask[0, : args["how_many"]] = 1
    if batch_size > 1:
        mask[1, : args["how_many"] - 1] = 1
    return mask


def transform(kspace=None, mask=None, **_kwargs):
    if isinstance(mask, torch.Tensor):
        mask = mask.view(mask.shape[0], 1, -1, 1)
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
        batch_size,
        loops_train,
        num_train=1,
        num_val=1,
        num_test=1,
        score_fn=None,
        seed=None,
    ):
        super().__init__(32, 64, batch_size=batch_size, seed=seed)
        self._num_loops_train_data = loops_train
        self._init_from_config_dict(config_dict)

        self._tensor_size = self._cfg["mask"]["args"]["size"]

        data_init_fn = make_data_init_fn(
            self._tensor_size, num_train, num_val, num_test
        )
        self._setup_data_handlers(data_init_fn)
        self._score_fn = score_fn

    def _compute_score_given_tensors(
        self, reconstruction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        if self._score_fn:
            return self._score_fn(reconstruction, ground_truth)

    def score_keys(self) -> List[str]:
        pass
