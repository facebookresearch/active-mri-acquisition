# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict, List


class Policy:
    """ A basic policy interface. """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_action(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        """ Returns a list of actions for a batch of observations. """
        pass

    def __call__(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        return self.get_action(obs, **kwargs)


from .simple_baselines import (
    RandomPolicy,
    RandomLowBiasPolicy,
    LowestIndexPolicy,
    OneStepGreedyOracle,
)
from .cvpr19_evaluator import CVPR19Evaluator
from .ddqn import DDQN, DDQNTrainer
from .evaluation import evaluate

__all__ = [
    "RandomPolicy",
    "RandomLowBiasPolicy",
    "LowestIndexPolicy",
    "OneStepGreedyOracle",
    "CVPR19Evaluator",
    "DDQN",
    "DDQNTrainer",
    "evaluate",
]
