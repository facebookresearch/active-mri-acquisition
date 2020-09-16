from typing import Any, Dict, List, Optional

import numpy as np
import torch

import activemri.envs
from . import Policy


class RandomPolicy(Policy):
    """ A policy representing random k-space selection.

        Returns one of the valid actions uniformly at random.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """ Returns a random action without replacement. """
        return (
            (obs["mask"].logical_not().float() + 1e-6).multinomial(1).squeeze().tolist()
        )


class LowestIndexPolicy(Policy):
    """ A policy that represents low-to-high frequency k-space selection.

        Args:
            alternate_sides(bool): If ``True`` the indices of selected actions will move
                    symmetrically towards the center. For example, for an image with 100
                    columns, the order will be 0, 99, 1, 98, 2, 97, ...

        Returns the lowest inactive column in the mask, corresponding to the lowest
        frequency assuming high frequencies are in the center of k-space.
    """

    def __init__(self, alternate_sides: bool):
        super().__init__()
        self.alternate_sides = alternate_sides
        self.bottom_side = True

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """ Returns a random action without replacement. """
        mask = obs["mask"]
        if self.bottom_side:
            idx = torch.arange(mask.shape[1], 0, -1)
        else:
            idx = torch.arange(mask.shape[1])
        if self.alternate_sides:
            self.bottom_side = not self.bottom_side
        # Next line finds the first non-zero index (from edge to center) and returns it
        return (mask.logical_not() * idx).argmax(dim=1).int().tolist()


class OneStepGreedyOracle(Policy):
    def __init__(
        self,
        env: activemri.envs.ActiveMRIEnv,
        metric: str,
        num_samples: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        assert metric in env.score_keys()
        super().__init__()
        self.env = env
        self.metric = metric
        self.num_samples = num_samples
        self.rng = rng if rng is not None else np.random.RandomState()

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """ Returns a random action without replacement. """
        mask = obs["mask"]
        batch_size = mask.shape[0]
        all_action_lists = []
        for i in range(batch_size):
            available_actions = mask[i].logical_not().nonzero().squeeze().tolist()
            self.rng.shuffle(available_actions)
            if len(available_actions) < self.num_samples:
                # Add dummy actions to try if num of samples is higher than the
                # number of inactive columns in this mask
                available_actions.extend(
                    [0] * (self.num_samples - len(available_actions))
                )
            all_action_lists.append(available_actions)

        all_scores = np.zeros((batch_size, self.num_samples))
        for i in range(self.num_samples):
            batch_action_to_try = [action_list[i] for action_list in all_action_lists]
            obs, new_score = self.env.try_action(batch_action_to_try)
            all_scores[:, i] = new_score[self.metric]
        if self.metric in ["mse", "nmse"]:
            all_scores *= -1
        else:
            assert self.metric in ["ssim", "psnr"]

        best_indices = all_scores.argmax(axis=1)
        action = []
        for i in range(batch_size):
            action.append(all_action_lists[i][best_indices[i]])
        return action
