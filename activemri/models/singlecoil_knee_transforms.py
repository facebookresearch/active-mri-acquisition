from typing import Tuple

import fastmri
import fastmri.data.transforms as fastmri_transforms
import torch

import activemri.data.singlecoil_knee_data as scknee_data


def dicom_to_0_1_range(tensor: torch.Tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def dicom_transform(image: torch.Tensor, mean: float, std: float):
    image = (image - mean) / (std + 1e-12)
    image = torch.from_numpy(image)
    image = dicom_to_0_1_range(image)
    image = torch.cat([image, torch.zeros_like(image)], dim=0)
    return image


def to_magnitude(tensor: torch.Tensor, dim: int):
    return (tensor ** 2).sum(dim=dim, keepdim=True) ** 0.5


def center_crop(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    # Expects tensor to be (bs, H, W, C) and shape to be 2-dim
    assert len(x.shape) == 4
    assert 0 < shape[0] <= x.shape[1]
    assert 0 < shape[1] <= x.shape[2]
    h_from = (x.shape[1] - shape[0]) // 2
    w_from = (x.shape[2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[:, h_from:h_to, w_from:w_to, :]
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
    # alter mask to always include the highest frequencies that include padding
    mask[
        :,
        scknee_data.MICCAI2020Data.START_PADDING : scknee_data.MICCAI2020Data.END_PADDING,
    ] = 1
    mask = mask.view(mask.shape[0], 1, 1, -1)

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

    # TODO remove this part once mask function is changed to return correct shape
    num_cols = kspace.shape[-2]
    mask = mask[:num_cols]  # accounting for variable size masks
    mask_shape = [1 for _ in kspace.shape]
    mask_shape[-2] = num_cols
    mask = mask.view(*mask_shape)
    masked_kspace = kspace * mask + 0.0

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
    image, *_ = fastmri_transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    return image.unsqueeze(0)


def _batched_fastmri_unet_transform(
    kspace, mask, ground_truth, attrs, which_challenge="singlecoil"
):
    batch_size = len(kspace)
    images = []
    for i in range(batch_size):
        image = _base_fastmri_unet_transform(
            kspace[i],
            mask[i],
            ground_truth[i],
            attrs[i],
            which_challenge=which_challenge,
        )
        images.append(image)
    return tuple([torch.stack(images)])  # environment expects a tuple


# noinspection PyUnusedLocal
def fastmri_unet_transform_singlecoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "singlecoil"
    )


# noinspection PyUnusedLocal
def fastmri_unet_transform_multicoil(
    kspace=None, mask=None, ground_truth=None, attrs=None, fname=None, slice_id=None
):
    return _batched_fastmri_unet_transform(
        kspace, mask, ground_truth, attrs, "multicoil"
    )
