import numpy as np

# noinspection PyUnresolvedReferences
import pytest
import torch

import activemri.envs.masks as masks


# noinspection PyCallingNonCallable,PyUnresolvedReferences
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
    mask_args = {
        "max_width": 20,
        "width_dim": 1,
        "min_cols": 1,
        "max_cols": 4,
    }
    rng = np.random.RandomState()
    widths = [10, 12, 18]
    seen_cols = set()
    for i in range(1000):
        dummy_shapes = [(0, w) for w in widths]  # w is in args.width_dim
        the_masks = masks.sample_low_frequency_mask(mask_args, dummy_shapes, rng)
        assert the_masks.shape == (len(widths), 20)

        for j, w in enumerate(widths):
            assert torch.all(
                the_masks[j, : w // 2] == torch.flip(the_masks[j, w // 2 : w], dims=[0])
            )
            assert the_masks[j, w:].sum().item() == 0
            active = the_masks[j].sum().int().item()
            assert active >= 2 * mask_args["min_cols"]
            assert active <= 2 * mask_args["max_cols"]
            assert active % 2 == 0
            seen_cols.add(active // 2)
    assert len(seen_cols) == (mask_args["max_cols"] - mask_args["min_cols"] + 1)
