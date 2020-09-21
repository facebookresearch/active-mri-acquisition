"""
activemri.data.transforms.py
====================================
Transform functions to process fastMRI data for reconstruction models.
"""
from typing import Tuple, Union

import fastmri
import fastmri.data.transforms as fastmri_transforms
import numpy as np
import torch

import activemri.data.singlecoil_knee_data as scknee_data

TensorType = Union[np.ndarray, torch.Tensor]


def dicom_to_0_1_range(tensor: torch.Tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def dicom_transform(image: torch.Tensor, mean: float, std: float):
    image = (image - mean) / (std + 1e-12)
    image = torch.from_numpy(image)
    image = dicom_to_0_1_range(image)
    image = torch.cat([image, torch.zeros_like(image)], dim=0)
    return image


def center_crop(x: TensorType, shape: Tuple[int, int]) -> TensorType:
    """ Center crops a tensor to the desired 2D shape.

        Args:
            x(union(``torch.Tensor``, ``np.ndarray``)): The tensor to crop.
                Shape should be ``(batch_size, height, width)``.
            shape(tuple(int,int)): The desired shape to crop to.

        Returns:
            (union(``torch.Tensor``, ``np.ndarray``)): The cropped tensor.
    """
    assert len(x.shape) == 3
    assert 0 < shape[0] <= x.shape[1]
    assert 0 < shape[1] <= x.shape[2]
    h_from = (x.shape[1] - shape[0]) // 2
    w_from = (x.shape[2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[:, h_from:h_to, w_from:w_to]
    return x


def ifft_permute_maybe_shift(
    x: torch.Tensor, normalized: bool = False, ifft_shift: bool = False
) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1)
    y = torch.ifft(x, 2, normalized=normalized)
    if ifft_shift:
        y = fastmri.ifftshift(y, dim=(1, 2))
    return y.permute(0, 3, 1, 2)


def raw_transform_miccai2020(kspace=None, mask=None, **_kwargs):
    """ Transform to produce input for reconstructor used in Pineda et al. MICCAI'20.

        Produces a zero-filled reconstruction and a mask that serve as a input to models of type
        :class:`activemri.models.cvpr10_reconstructor.CVPR19Reconstructor`. The mask is almost
        equal to the mask passed as argument, except that high-frequency padding columns are set
        to 1, and the mask is reshaped to be compatible with the reconstructor.

        Args:
            kspace(``np.ndarray``): The array containing the k-space data returned by the dataset.
            mask(``torch.Tensor``): The masks to apply to the k-space.

        Returns:
            tuple: A tuple containing:
                - ``torch.Tensor``: The zero-filled reconstructor that will be passed to the
                  reconstructor.
                - ``torch.Tensor``: The mask to use as input to the reconstructor.
    """
    # alter mask to always include the highest frequencies that include padding
    mask[
        :,
        :,
        scknee_data.MICCAI2020Data.START_PADDING : scknee_data.MICCAI2020Data.END_PADDING,
    ] = 1
    mask = mask.unsqueeze(1)

    all_kspace = []
    for ksp in kspace:
        all_kspace.append(torch.from_numpy(ksp).permute(2, 0, 1))
    k_space = torch.stack(all_kspace)

    masked_true_k_space = torch.where(
        mask.byte(), k_space, torch.tensor(0.0).to(mask.device),
    )
    reconstructor_input = ifft_permute_maybe_shift(masked_true_k_space, ifft_shift=True)
    return reconstructor_input, mask


# Based on
# https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py
def _base_fastmri_unet_transform(
    kspace, mask, ground_truth, attrs, which_challenge="singlecoil",
):
    kspace = fastmri_transforms.to_tensor(kspace)

    mask = mask[..., : kspace.shape[-2]]  # accounting for variable size masks
    masked_kspace = kspace * mask.unsqueeze(-1) + 0.0

    # inverse Fourier transform to get zero filled solution
    image = fastmri.ifft2c(masked_kspace)

    # crop input to correct size
    if ground_truth is not None:
        crop_size = (ground_truth.shape[-2], ground_truth.shape[-1])
    else:
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

    # check for FLAIR 203
    if image.shape[-2] < crop_size[1]:
        crop_size = (image.shape[-2], image.shape[-2])

    # noinspection PyTypeChecker
    image = fastmri_transforms.complex_center_crop(image, crop_size)

    # absolute value
    image = fastmri.complex_abs(image)

    # apply Root-Sum-of-Squares if multicoil data
    if which_challenge == "multicoil":
        image = fastmri.rss(image)

    # normalize input
    image, mean, std = fastmri_transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    return image.unsqueeze(0), mean, std


def _batched_fastmri_unet_transform(
    kspace, mask, ground_truth, attrs, which_challenge="singlecoil"
):
    batch_size = len(kspace)
    images, means, stds = [], [], []
    for i in range(batch_size):
        image, mean, std = _base_fastmri_unet_transform(
            kspace[i],
            mask[i],
            ground_truth[i],
            attrs[i],
            which_challenge=which_challenge,
        )
        images.append(image)
        means.append(mean)
        stds.append(std)
    return torch.stack(images), torch.stack(means), torch.stack(stds)


# noinspection PyUnusedLocal
def fastmri_unet_transform_singlecoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    """ Transform to use as input to fastMRI's Unet model for singlecoil data.

        This is an adapted version of the code found in
        `fastMRI <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py#L190>`_.
    """
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "singlecoil"
    )


# noinspection PyUnusedLocal
def fastmri_unet_transform_multicoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    """ Transform to use as input to fastMRI's Unet model for multicoil data.

        This is an adapted version of the code found in
        `fastMRI <https://github.com/facebookresearch/fastMRI/blob/master/experimental/unet/unet_module.py#L190>`_.
    """
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "multicoil"
    )
