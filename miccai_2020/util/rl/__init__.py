from . import dqn
from . import evaluation
from . import replay_buffer
from . import simple_baselines

__all__ = ["dqn", "evaluation", "replay_buffer", "simple_baselines"]


class Policy:
    """ A basic policy interface. """

    def __init__(self):
        pass

    def get_action(self, *args):
        pass

    def init_episode(self):
        pass
