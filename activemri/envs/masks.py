from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def update_masks_from_indices(
    masks: torch.Tensor, indices: Sequence[int]
) -> torch.Tensor:
    assert masks.shape[0] == len(indices)
    for i in range(len(indices)):
        masks[i, ..., indices[i]] = 1
    return masks


def check_masks_complete(masks: torch.Tensor) -> List[bool]:
    done = []
    for mask in masks:
        done.append(mask.bool().all().item())
    return done


def sample_low_frequency_mask(
    mask_args: Dict[str, Any], size: int, rng: np.random.RandomState
) -> torch.Tensor:
    mask = torch.zeros(size, mask_args["img_width"])
    num_lowf = rng.randint(mask_args["min_cols"], mask_args["max_cols"] + 1, size=size)
    for i in range(size):
        mask[i, : num_lowf[i]] = mask[i, -num_lowf[i] :] = 1
    return mask
