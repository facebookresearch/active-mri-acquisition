import torch

import miccai_2020.models.fft_utils


def dicom_to_0_1_range(tensor: torch.Tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def dicom_transform(image: torch.Tensor, mean: float, std: float):
    image = (image - mean) / (std + 1e-12)
    image = torch.from_numpy(image)
    image = dicom_to_0_1_range(image)
    image = torch.cat([image, torch.zeros_like(image)], dim=0)
    return image


def to_magnitude(tensor):
    tensor = (tensor[:, 0, :, :] ** 2 + tensor[:, 1, :, :] ** 2) ** 0.5
    return tensor.unsqueeze(1)


def raw_transform_miccai20(kspace, mask, target):
    k_space = kspace.permute(0, 3, 1, 2)
    # alter mask to always include the highest frequencies that include padding
    mask = torch.where(
        to_magnitude(k_space).sum(2).unsqueeze(2) == 0.0, torch.tensor(1.0), mask,
    )
    masked_true_k_space = torch.where(mask.byte(), k_space, torch.tensor(0.0))
    reconstructor_input = miccai_2020.models.fft_utils.ifft(
        masked_true_k_space, ifft_shift=True
    )
    target = target.permute(0, 3, 1, 2)
    return reconstructor_input, target, mask
