import abc
from typing import Any, Dict, List


class Policy:
    """ A basic policy interface. """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_action(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        pass

    def __call__(self, obs: Dict[str, Any], **kwargs: Any) -> List[int]:
        return self.get_action(obs, **kwargs)


from .simple_baselines import RandomPolicy, LowestIndexPolicy
from .cvpr19_evaluator import CVPR19Evaluator
from .ddqn import DDQN, DDQNTrainer
from .evaluation import evaluate

__all__ = [
    "RandomPolicy",
    "LowestIndexPolicy",
    "CVPR19Evaluator",
    "DDQN",
    "DDQNTrainer",
    "evaluate",
]
