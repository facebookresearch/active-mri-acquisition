import torch

from typing import Any, Dict, Optional

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
