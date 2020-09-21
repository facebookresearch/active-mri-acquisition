"""
activemri.baselines.simple_baselines.py
=======================================
Simple baselines for active MRI acquisition.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import activemri.envs
from . import Policy


class RandomPolicy(Policy):
    """ A policy representing random k-space selection.

        Returns one of the valid actions uniformly at random.

        Args:
            seed(optional(int)): The seed to use for the random number generator, which is
                based on ``torch.Generator()``.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = torch.Generator()
        if seed:
            self.rng.manual_seed(seed)

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """ Returns a random action without replacement.

            Args:
                obs(dict(str, any)): As returned by :class:`activemri.envs.ActiveMRIEnv`.

            Returns:
                list(int): A list of random k-space column indices, one per batch element in
                    the observation. The indices are sampled from the set of inactive (0) columns
                    on each batch element.
        """
        return (
            (obs["mask"].logical_not().float() + 1e-6)
            .multinomial(1, generator=self.rng)
            .squeeze()
            .tolist()
        )


class RandomLowBiasPolicy(Policy):
    def __init__(
        self, acceleration: float, centered: bool = True, seed: Optional[int] = None
    ):
        super().__init__()
        self.acceleration = acceleration
        self.centered = centered
        self.rng = np.random.RandomState(seed)

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        mask = obs["mask"].squeeze().cpu().numpy()
        new_mask = self._cartesian_mask(mask)
        action = (new_mask - mask).argmax(axis=1)
        return action.tolist()

    @staticmethod
    def _normal_pdf(length: int, sensitivity: float):
        return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

    def _cartesian_mask(self, current_mask: np.ndarray) -> np.ndarray:
        batch_size, image_width = current_mask.shape
        pdf_x = RandomLowBiasPolicy._normal_pdf(
            image_width, 0.5 / (image_width / 10.0) ** 2
        )
        pdf_x = np.expand_dims(pdf_x, axis=0)
        lmda = image_width / (2.0 * self.acceleration)
        # add uniform distribution
        pdf_x += lmda * 1.0 / image_width
        # remove previously chosen columns
        # note that pdf_x designed for centered masks
        new_mask = (
            np.fft.ifftshift(current_mask, axes=1)
            if not self.centered
            else current_mask.copy()
        )
        pdf_x = pdf_x * np.logical_not(new_mask)
        # normalize probabilities and choose accordingly
        pdf_x /= pdf_x.sum(axis=1, keepdims=True)
        indices = [
            self.rng.choice(image_width, 1, False, pdf_x[i]).item()
            for i in range(batch_size)
        ]
        new_mask[range(batch_size), indices] = 1
        if not self.centered:
            new_mask = np.fft.ifftshift(new_mask, axes=1)
        return new_mask


class LowestIndexPolicy(Policy):
    """ A policy that represents low-to-high frequency k-space selection.

        Args:
            alternate_sides(bool): If ``True`` the indices of selected actions will alternate
                between the sides of the mask. For example, for an image with 100
                columns, and non-centered k-space, the order will be 0, 99, 1, 98, 2, 97, ..., etc.
                For the same size and centered, the order will be 49, 50, 48, 51, 47, 52, ..., etc.

            centered(bool): If ``True`` (default), low frequencies are in the center of the mask.
                Otherwise, they are in the edges of the mask.
    """

    def __init__(
        self, alternate_sides: bool, centered: bool = True,
    ):
        super().__init__()
        self.alternate_sides = alternate_sides
        self.centered = centered
        self.bottom_side = True

    def get_action(self, obs: Dict[str, Any], **_kwargs) -> List[int]:
        """ Returns a random action without replacement.

            Args:
                obs(dict(str, any)): As returned by :class:`activemri.envs.ActiveMRIEnv`.

            Returns:
                list(int): A list of k-space column indices, one per batch element in
                    the observation, equal to the lowest non-active k-space column in their
                    corresponding observation masks.
        """
        mask = obs["mask"].squeeze().cpu().numpy()
        new_mask = self._get_new_mask(mask)
        action = (new_mask - mask).argmax(axis=1)
        return action.tolist()

    def _get_new_mask(self, current_mask: np.ndarray) -> np.ndarray:
        # The code below assumes mask in non centered
        new_mask = (
            np.fft.ifftshift(current_mask, axes=1)
            if self.centered
            else current_mask.copy()
        )
        if self.bottom_side:
            idx = np.arange(new_mask.shape[1], 0, -1)
        else:
            idx = np.arange(new_mask.shape[1])
        if self.alternate_sides:
            self.bottom_side = not self.bottom_side
        # Next line finds the first non-zero index (from edge to center) and returns it
        indices = (np.logical_not(new_mask) * idx).argmax(axis=1)
        indices = np.expand_dims(indices, axis=1)
        new_mask[range(new_mask.shape[0]), indices] = 1
        if self.centered:
            new_mask = np.fft.ifftshift(new_mask, axes=1)
        return new_mask


class OneStepGreedyOracle(Policy):
    """ A policy that returns the k-space column leading to best reconstruction score.

        Args:
            env(``activemri.envs.ActiveMRIEnv``): The environment for which the policy is computed
                for.
            metric(str): The name of the score metric to use (must be in ``env.score_keys()``).
            num_samples(optional(int)): If given, only ``num_samples`` random actions will be
                tested. Defaults to ``None``, which means that method will consider all actions.
            rng(``numpy.random.RandomState``): A random number generator to use for sampling.
    """

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
        """ Returns a one-step greedy action maximizing reconstruction score.

            Args:
                obs(dict(str, any)): As returned by :class:`activemri.envs.ActiveMRIEnv`.

            Returns:
                list(int): A list of k-space column indices, one per batch element in
                    the observation, equal to the action that maximizes reconstruction score
                    (e.g, SSIM or negative MSE).
        """
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
