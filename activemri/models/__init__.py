# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict, Optional

import torch.nn


class Reconstructor(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def init_from_checkpoint(self, checkpoint: Dict[str, Any]) -> Optional[Any]:
        pass
