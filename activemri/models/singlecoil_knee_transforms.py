import fastMRI.data.transforms as fastmri_transforms
import torch


def dicom_to_0_1_range(tensor: torch.Tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def dicom_transform(image: torch.Tensor, mean: float, std: float):
    image = (image - mean) / (std + 1e-12)
    image = torch.from_numpy(image)
    image = dicom_to_0_1_range(image)
    image = torch.cat([image, torch.zeros_like(image)], dim=0)
    return image


def to_magnitude(tensor):
    tensor = (tensor[:, 0] ** 2 + tensor[:, 1] ** 2) ** 0.5
    return tensor.unsqueeze(0)


def ifft_permute_maybe_shift(x, normalized=False, ifft_shift=False):
    x = x.permute(0, 2, 3, 1)
    y = torch.ifft(x, 2, normalized=normalized)
    if ifft_shift:
        y = fastmri_transforms.ifftshift(y, dim=(1, 2))
    return y.permute(0, 3, 1, 2)


def raw_transform_miccai20(kspace, mask, *_):
    k_space = kspace.permute(0, 3, 1, 2)
    # alter mask to always include the highest frequencies that include padding
    mask = torch.where(
        to_magnitude(k_space).sum(2, keepdim=True) == 0.0,
        torch.tensor(1.0).to(mask.device),
        mask,
    )
    masked_true_k_space = torch.where(
        mask.byte(), k_space, torch.tensor(0.0).to(mask.device)
    )
    reconstructor_input = ifft_permute_maybe_shift(masked_true_k_space, ifft_shift=True)
    return reconstructor_input, mask
