import abc

from typing import Any, Dict, Optional

import torch
import torch.nn


# noinspection PyUnusedLocal,PyAbstractClass,PyMethodMayBeStatic
class Reconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def init_from_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        pass
