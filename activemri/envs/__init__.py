import numpy as np
import torch

from typing import Any, Dict, Optional

from .envs import (
    ActiveMRIEnv,
    CyclicSampler,
    DataHandler,
    SingleCoilKneeRAWEnv,
)

__all__ = [
    "ActiveMRIEnv",
    "CyclicSampler",
    "DataHandler",
    "SingleCoilKneeRAWEnv",
    "Reconstructor",
]


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


# noinspection PyUnusedLocal
def transform(kspace: torch.Tensor, maak: torch.Tensor, target: torch.Tensor, *args):
    pass


# noinspection PyUnusedLocal
def mask_function(mask_cfg: Dict[str, Any], rng: np.random.RandomState) -> np.ndarray:
    pass
