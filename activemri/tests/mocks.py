import numpy as np
import torch

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
            "how_many":3
        }
    },
    "reward_metric": "ssim",
    "device": "cpu"
}
"""


class Dataset:
    size = 10

    def __len__(self):
        return 2

    def __getitem__(self, item):
        mock_kspace = (item + 1) * np.ones((self.size, self.size, 2))
        mock_mask = np.zeros(self.size)
        mock_ground_truth = mock_kspace + 1
        return mock_kspace, mock_mask, mock_ground_truth, {}, "fname", item


def mask_func(args, batch_size, _rng):
    mask = torch.zeros(batch_size, Dataset.size)
    mask[0, : args["how_many"]] = 1
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
