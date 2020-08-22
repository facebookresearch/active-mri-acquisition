from typing import Tuple

import fastmri
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


def raw_transform_miccai20(kspace=None, mask=None, **_kwargs):
    k_space = kspace.permute(0, 3, 1, 2)
    # alter mask to always include the highest frequencies that include padding
    mask[
        :, scknee_data.RawSliceData.START_PADDING : scknee_data.RawSliceData.END_PADDING
    ] = 1
    mask = mask.view(mask.shape[0], 1, 1, -1)
    masked_true_k_space = torch.where(
        mask.byte(), k_space, torch.tensor(0.0).to(mask.device),
    )
    reconstructor_input = ifft_permute_maybe_shift(masked_true_k_space, ifft_shift=True)
    return reconstructor_input, mask
