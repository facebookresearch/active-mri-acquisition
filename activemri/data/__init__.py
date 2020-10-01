# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import numpy as np
import torch

from . import singlecoil_knee_data
from . import transforms

__all__ = ["singlecoil_knee_data", "transforms"]


def transform_template(
    kspace: List[np.ndarray] = None,
    mask: torch.Tensor = None,
    ground_truth: torch.Tensor = None,
    attrs: List[Dict[str, Any]] = None,
    fname: List[str] = None,
    slice_id: List[int] = None,
):
    """Template for transform functions.

    Args:
        - kspace(list(np.ndarray)): A list of complex numpy arrays, one per k-space in the batch.
          The length is the ``batch_size``, and array shapes are ``H x W x 2`` for single coil data,
          and ``C x H x W x 2`` for multicoil data, where ``H`` denotes k-space height, ``W``
          denotes k-space width, and ``C`` is the number of coils. Note that the width can differ
          between batch elements, if ``num_cols`` is set to a tuple when creating the environment.
        - mask(torch.Tensor): A tensor of binary column masks, where 1s indicate that the
          corresponding k-space column should be selected. The shape is ``batch_size x 1 x maxW``,
          for single coil data, and ``batch_size x 1 x 1 x maxW`` for multicoil data. Here ``maxW``
          is the maximum k-space width returned by the environment.
        - ground_truth(torch.Tensor): A tensor of ground truth 2D images. The shape is
         ``batch_size x 320 x 320``.
        - attrs(list(dict)): A list of dictionaries with the attributes read from the fastMRI for
          each image.
        - fname(list(str)): A list of the filenames where the images where read from.
        - slice_id(list(int)): A list with the slice ids in the files where each image was read
          from.

    Returns:
        tuple(Any...): A tuple with any number of inputs required by the reconstructor model.

    """
    pass
