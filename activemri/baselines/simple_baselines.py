from typing import Any, Dict, List

import torch

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
