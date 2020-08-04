from typing import Any, Dict

import numpy as np
import torch


def update_masks_from_indices(masks: torch.Tensor, indices: np.ndarray):
    assert masks.shape[0] == indices.size
    for i, index in enumerate(indices):
        masks[i, :, index] = 1
    return masks


def sample_low_frequency_mask(
    mask_args: Dict[str, Any], rng: np.random.RandomState
) -> torch.Tensor:
    mask = torch.zeros(mask_args["img_width"])
    num_lowf = rng.randint(mask_args["min_cols"], mask_args["max_cols"] + 1)
    mask[:num_lowf] = mask[-num_lowf:] = 1
    return mask
