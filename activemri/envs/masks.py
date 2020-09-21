"""
activemri.envs.masks.py
====================================
Utilities to generate and manipulate active acquisition masks.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fastmri
import numpy as np
import torch


def update_masks_from_indices(
    masks: torch.Tensor, indices: Sequence[int]
) -> torch.Tensor:
    assert masks.shape[0] == len(indices)
    new_masks = masks.clone()
    for i in range(len(indices)):
        new_masks[i, ..., indices[i]] = 1
    return new_masks


def check_masks_complete(masks: torch.Tensor) -> List[bool]:
    done = []
    for mask in masks:
        done.append(mask.bool().all().item())
    return done


# TODO remove width dim arg, and just have the mask determine width from last kspace dimension
#   Then make the return shape be equal to [len(kspace_shape) + 1] dims.
def sample_low_frequency_mask(
    mask_args: Dict[str, Any],
    kspace_shapes: List[Tuple[int, ...]],
    rng: np.random.RandomState,
    attrs: Optional[List[Dict[str, Any]]] = None,
) -> torch.Tensor:
    """ Samples low frequency masks.

        Returns masks that contain some number of the lowest k-space frequencies active.
        The number of frequencies doesn't have to be the same for all masks in the batch, and
        it can also be a random number, depending on the given ``mask_args``. Active columns
        will be represented as 1s in the mask, and inactive columns as 0s.

        The distribution and shape of the masks can be controlled by ``mask_args``. This is a
        dictionary with the following keys:

            - *"max_width"(int)*: The maximum width of the masks.
            - *"min_cols"(int)*: The minimum number of low frequencies columns to activate.
            - *"max_cols"(int)*: The maximum number of low frequencies columns to activate
              (inclusive).
            - *"width_dim"(int)*: Indicates which of the dimensions in ``kspace_shapes``
              corresponds to the k-space width.
            - *"centered"(bool)*: Specifies if the low frequencies are in the center of the
              k-space (``True``) or on the edges (``False``).
            - *"apply_attrs_padding"(optional(bool))*: If ``True``, the function will read
              keys ``"padding_left"`` and ``"padding_right"`` from ``attrs`` and set all
              corresponding high-frequency columns to 1.

        The number of 1s in the effective region of the mask (see next paragraph) is sampled
        between ``mask_args["min_cols"]`` and ``mask_args["max_cols"]`` (inclusive).
        The number of dimensions for the mask tensor will be ``mask_args["width_dim"] + 2``.
        The size will be ``[batch_size, 1, ..., 1, mask_args["max_width"]]``. For example, with
        ``mask_args["width_dim"] = 1`` and ``mask_args["max_width"] = 368``, output tensor
        has shape ``[batch_size, 1, 368]``.

        This function supports simultaneously sampling masks for k-space of different number of
        columns. This is controlled by argument ``kspace_shapes``. From this list, the function will
        obtain 1) ``batch_size = len(kspace_shapes``), and 2) the width of the k-spaces for
        each element in the batch. The i-th mask will have
        ``kspace_shapes[item][mask_args["width_dim"]]``
        *effective* columns.


        Note:
            The mask tensor returned will always have
            ``mask_args["max_width"]`` columns. However, for any element ``i``
            s.t.  ``kspace_shapes[i][mask_args["width_dim"]] < mask_args["max_width"]``, the
            function will then pad the extra k-space columns with 1s. The rest of the columns
            will be filled out as if the mask has the same width as that indicated by
            ``kspace_shape[i]``.

        Args:
            mask_args(dict(str,any)): Specifies configuration options for the masks, as explained
                above.

            kspace_shapes(list(tuple(int,...))): Specifies the shapes of the k-space data on
                which this mask will be applied, as explained above.

            rng(``np.random.RandomState``): A random number generator to sample the masks.

            attrs(dict(str,int)): Used to determine any high-frequency padding. It must contain
                keys ``"padding_left"`` and ``"padding_right"``.

        Returns:
            ``torch.Tensor``: The generated low frequency masks.

    """
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
