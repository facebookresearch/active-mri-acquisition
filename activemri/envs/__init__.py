import numpy as np
import pathlib
import torch

from typing import Any, Dict, List, Optional, Tuple

__all__ = ["SingleCoilKneeRAWEnv"]

from .envs import SingleCoilKneeRAWEnv


# noinspection PyUnusedLocal,PyAbstractClass
class Reconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)

    def forward(
        self, zero_filled_image: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> Dict[str, Any]:
        return {"reconstruction": zero_filled_image, "mask": mask}

    def load_from_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        pass


def transform(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    attrs: List[Dict[str, Any]],
    fname: List[pathlib.Path],
    slice_id: List[int],
) -> Tuple:
    return kspace, mask, target, attrs, fname, slice_id


# noinspection PyUnusedLocal
def mask_function(mask_cfg: Dict[str, Any], rng: np.random.RandomState) -> np.ndarray:
    pass
