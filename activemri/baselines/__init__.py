import abc
from typing import Any, Dict, List


class Policy:
    """ A basic policy interface. """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_action(self, obs: Dict[str, Any]) -> List[int]:
        pass

    @abc.abstractmethod
    def init_episode(self):
        pass

    def __call__(self, obs: Dict[str, Any]) -> List[int]:
        return self.get_action(obs)


from .simple_baselines import *

__all__ = ["RandomPolicy", "LowestIndexPolicy"]
