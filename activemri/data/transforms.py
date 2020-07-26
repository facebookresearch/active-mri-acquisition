import fastMRI.data.transforms as fastmri_transforms
import numpy as np
import torch


def raw_transform_miccai20(data, *_):
    kspace = torch.from_numpy(np.stack([data.real, data.imag], axis=-1))
    kspace = fastmri_transforms.ifftshift(kspace, dim=(0, 1))
    image = torch.ifft(kspace, 2, normalized=False)
    image = fastmri_transforms.ifftshift(image, dim=(0, 1))
    # Mean k-space of training data
    image /= 7.072103529760345e-07
    kspace /= 7.072103529760345e-07
    return kspace


def dicom_to_0_1_range(tensor):
    return (tensor.clamp(-3, 3) + 3) / 6


def dicom_transform(image, mean, std):
    image = (image - mean) / (std + 1e-12)
    image = torch.from_numpy(image)
    image = dicom_to_0_1_range(image)
    image = torch.cat([image, torch.zeros_like(image)], dim=0)
    return image
