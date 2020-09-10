from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# TODO change test so that it accounts for the padding
# TODO check the order of frequencies in fastMRI repo. I think low frequencies are in the center
# TODO change code so that mask returns same num of dims as kspace
def sample_low_frequency_mask(
    mask_args: Dict[str, Any],
    kspace_shapes: List[Tuple[int, ...]],
    rng: np.random.RandomState,
    attrs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    batch_size = len(kspace_shapes)
    num_cols = [shape[mask_args["width_dim"]] for shape in kspace_shapes]
    mask = torch.zeros(batch_size, mask_args["max_width"])
    num_lowf = rng.randint(
        mask_args["min_cols"], mask_args["max_cols"] + 1, size=batch_size
    )
    for i in range(batch_size):
        # If padding needs to be accounted for, only add low frequency lines
        # beyond the padding
        if attrs and mask_args.get("apply_attrs_padding", False):
            padding_left = attrs[i]["padding_left"]
            padding_right = attrs[i]["padding_right"]
        else:
            padding_left, padding_right = 0, num_cols[i]
        left_active = slice(0, num_lowf[i] + padding_left)
        right_active = slice(padding_right - num_lowf[i], mask_args["max_width"])
        mask[i, left_active] = 1
        mask[i, right_active] = 1
    return mask
