# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch


def get_mask_func(mask_type, which_dataset, rnl_params=None):
    # Whether the number of lines is random or not
    random_num_lines = mask_type[-4:] == "_rnl"
    if "symmetric_basic" in mask_type:
        logging.info(
            f"Mask is symmetric uniform choice with random_num_lines={random_num_lines}."
        )
        return SymmetricUniformChoiceMaskFunc(
            [0.125],
            [4],
            which_dataset,
            random_num_lines=random_num_lines,
            rnl_params=rnl_params,
        )
    if "basic" in mask_type:
        # First two parameters are ignored if `random_num_lines` is True
        logging.info(
            f"Mask is fixed acceleration mask with random_num_lines={random_num_lines}."
        )
        return BasicMaskFunc(
            [0.125],
            [4],
            which_dataset,
            random_num_lines=random_num_lines,
            rnl_params=rnl_params,
        )
    if "low_to_high" in mask_type:
        logging.info(
            f"Mask is symmetric low to high with random_num_lines={random_num_lines}."
        )
        return SymmetricLowToHighMaskFunc(
            [0.125],
            [4],
            which_dataset,
            random_num_lines=random_num_lines,
            rnl_params=rnl_params,
        )
    if "symmetric_grid" in mask_type:
        logging.info("Mask is symmetric grid.")
        return SymmetricUniformGridMaskFunc(
            [], [], which_dataset, random_num_lines=True, rnl_params=rnl_params
        )
    if "grid" in mask_type:
        logging.info("Mask is grid (not symmetric).")
        return UniformGridMaskFunc(
            [], [], which_dataset, random_num_lines=True, rnl_params=rnl_params
        )
    raise ValueError(f"Invalid mask type: {mask_type}.")


class MaskFunc:
    def __init__(
        self,
        center_fractions,
        accelerations,
        which_dataset,
        random_num_lines=False,
        rnl_params=None,
    ):
        if len(center_fractions) != len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.random_num_lines = random_num_lines

        if rnl_params is None:
            # The lines below give approx. 4x acceleration on average.
            self.min_lowf_lines = 10 if which_dataset != "KNEE_RAW" else 30
            self.max_lowf_lines = 12 if which_dataset != "KNEE_RAW" else 32
            self.highf_beta_alpha = 1
            self.highf_beta_beta = 5
        else:
            params = [int(x) for x in rnl_params.split(",")]
            assert len(params) == 4
            self.min_lowf_lines = params[0]
            self.max_lowf_lines = params[1]
            self.highf_beta_alpha = params[2]
            self.highf_beta_beta = params[3]

        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        self.rng.seed(seed)
        num_cols = shape[-2]

        # Determine number of low and high frequency lines to scan
        if self.random_num_lines:
            # These are guaranteed to be an even number (useful for symmetric masks)
            num_low_freqs = self.rng.choice(
                range(self.min_lowf_lines, self.max_lowf_lines, 2)
            )
            num_high_freqs = (
                int(
                    self.rng.beta(self.highf_beta_alpha, self.highf_beta_beta)
                    * (num_cols - num_low_freqs)
                    // 2
                )
                * 2
            )
        else:
            choice = self.rng.randint(0, len(self.accelerations))
            center_fraction = self.center_fractions[choice]
            acceleration = self.accelerations[choice]

            num_low_freqs = int(round(num_cols * center_fraction))
            num_high_freqs = int(num_cols // acceleration - num_low_freqs)

        # Create the mask
        mask = self.create_lf_focused_mask(num_cols, num_high_freqs, num_low_freqs)

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-1] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        p = num_high_freqs / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < p
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True
        return mask


class BasicMaskFunc(MaskFunc):
    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        hf_cols = self.rng.choice(
            np.arange(num_cols - num_low_freqs), num_high_freqs, replace=False
        )
        hf_cols[hf_cols >= (num_cols - num_low_freqs + 1) // 2] += num_low_freqs
        mask[hf_cols] = True
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad : pad + num_low_freqs] = True
        mask = np.fft.ifftshift(mask, axes=0)
        return mask


class SymmetricUniformChoiceMaskFunc(MaskFunc):
    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        num_cols //= 2
        num_low_freqs //= 2
        num_high_freqs //= 2
        hf_cols = self.rng.choice(
            np.arange(num_cols - num_low_freqs), num_high_freqs, replace=False
        )
        mask[hf_cols] = True
        pad = num_cols - num_low_freqs
        mask[pad:num_cols] = True
        mask[: -(num_cols + 1) : -1] = mask[:num_cols]
        mask = np.fft.ifftshift(mask, axes=0)
        return mask


class UniformGridMaskFunc(MaskFunc):
    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        acceleration = self.rng.choice([4, 8, 16])
        hf_cols = np.arange(acceleration, num_cols, acceleration)
        mask[hf_cols] = True
        mask[: num_low_freqs // 2] = mask[-(num_low_freqs // 2) :] = True
        return mask


class SymmetricLowToHighMaskFunc(MaskFunc):
    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        num_cols //= 2
        num_low_freqs //= 2
        num_high_freqs //= 2
        num_low_freqs += num_high_freqs
        pad = num_cols - num_low_freqs
        mask[pad:num_cols] = True
        mask[: -(num_cols + 1) : -1] = mask[:num_cols]
        mask = np.fft.ifftshift(mask, axes=0)
        return mask


class SymmetricUniformGridMaskFunc(MaskFunc):
    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        mask = np.zeros([num_cols])
        acceleration = self.rng.choice([4, 8, 16])
        num_cols //= 2
        num_low_freqs //= 2
        hf_cols = np.arange(acceleration, num_cols, acceleration)
        mask[hf_cols] = True
        mask[:num_low_freqs] = True
        mask[: -(num_cols + 1) : -1] = mask[:num_cols]
        return mask
