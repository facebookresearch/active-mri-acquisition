# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest  # noqa: F401
import torch

import activemri.envs.masks as masks


def test_update_masks_from_indices():
    mask_1 = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8)
    mask_2 = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.uint8)
    mask = torch.stack([mask_1, mask_2])
    mask = masks.update_masks_from_indices(mask, np.array([2, 0]))
    assert mask.shape == torch.Size([2, 3, 4])

    expected = torch.tensor(
        [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=torch.uint8
    ).repeat(2, 1, 1)
    assert (mask - expected).sum().item() == 0


def test_sample_low_freq_masks():
    for centered in [True, False]:
        max_width = 20
        mask_args = {
            "max_width": max_width,
            "width_dim": 1,
            "min_cols": 1,
            "max_cols": 4,
            "centered": centered,
        }
        rng = np.random.RandomState()
        widths = [10, 12, 18, 20]
        seen_cols = set()
        for i in range(1000):
            dummy_shapes = [(0, w) for w in widths]  # w is in args.width_dim
            the_masks = masks.sample_low_frequency_mask(mask_args, dummy_shapes, rng)
            assert the_masks.shape == (len(widths), 1, 20)
            the_masks = the_masks.squeeze()

            for j, w in enumerate(widths):
                # Mask is symmetrical
                assert torch.all(
                    the_masks[j, : w // 2]
                    == torch.flip(the_masks[j, w // 2 : w], dims=[0])
                )
                # Extra columns set to one so that they are not valid actions
                assert the_masks[j, w:].sum().item() == max_width - w

                # Check that the number of columns is in the correct range
                active = the_masks[j, :w].sum().int().item()
                assert active >= 2 * mask_args["min_cols"]
                assert active <= 2 * mask_args["max_cols"]
                seen_cols.add(active // 2)

                # These masks should be either something like
                # 1100000011|111111111 (not centered)
                # 0000110000|111111111 (centered)
                # The lines below check for this
                prev = the_masks[j, 0]
                changed = False
                for k in range(1, w // 2):
                    cur = the_masks[j, k]
                    if cur != prev:
                        assert not changed
                        changed = True
                    prev = cur
                assert changed
                if centered:
                    assert not the_masks[j, 0]
                else:
                    assert the_masks[j, 0]

        # Check that masks were sampled with all possible number of active cols
        assert len(seen_cols) == (mask_args["max_cols"] - mask_args["min_cols"] + 1)
