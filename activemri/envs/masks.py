from typing import Any, Dict, List, Optional, Sequence, Tuple

import fastmri
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
    mask_args: Dict[str, Any],
    kspace_shapes: List[Tuple[int, ...]],
    rng: np.random.RandomState,
    attrs: Optional[List[Dict[str, Any]]] = None,
) -> torch.Tensor:
    batch_size = len(kspace_shapes)
    num_cols = [shape[mask_args["width_dim"]] for shape in kspace_shapes]
    mask = torch.zeros(batch_size, mask_args["max_width"])
    num_low_freqs = rng.randint(
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

        pad = (num_cols[i] - 2 * num_low_freqs[i] + 1) // 2
        mask[i, pad : pad + 2 * num_low_freqs[i]] = 1
        mask[i, :padding_left] = 1
        mask[i, padding_right : num_cols[i]] = 1

        if not mask_args["centered"]:
            mask[i, : num_cols[i]] = fastmri.ifftshift(mask[i, : num_cols[i]])
        mask[i, num_cols[i] : mask_args["max_width"]] = 1

    mask_shape = [batch_size] + [1] * (mask_args["width_dim"] + 1)
    mask_shape[mask_args["width_dim"] + 1] = mask_args["max_width"]
    return mask.view(*mask_shape)
