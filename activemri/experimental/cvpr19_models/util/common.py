# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
import torchvision.utils as tvutil


def load_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    if os.path.isfile(checkpoint_path):
        logging.info(f"Found checkpoint at {checkpoint_path}.")
        return torch.load(checkpoint_path)
    logging.info(f"No checkpoint found at {checkpoint_path}.")
    return None


def compute_ssims(xs, ys):
    ssims = []
    for i in range(xs.shape[0]):
        ssim = skimage.measure.compare_ssim(
            xs[i, 0].cpu().numpy(),
            ys[i, 0].cpu().numpy(),
            data_range=ys[i, 0].cpu().numpy().max(),
        )
        ssims.append(ssim)
    return np.array(ssims).mean()


def compute_psnrs(xs, ys):
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = skimage.measure.compare_psnr(
            xs[i, 0].cpu().numpy(),
            ys[i, 0].cpu().numpy(),
            data_range=ys[i, 0].cpu().numpy().max(),
        )
        psnrs.append(psnr)
    return np.array(psnrs).mean()


def compute_mse(xs, ys):
    return np.mean((ys.cpu().numpy() - xs.cpu().numpy()) ** 2)


def compute_nmse(xs, ys):
    ys_numpy = ys.cpu().numpy()
    return (
        np.linalg.norm(ys_numpy - xs.cpu().numpy()) ** 2 / np.linalg.norm(ys_numpy) ** 2
    )


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8, renormalize=True):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image

    # do normalization first, since we working on fourier space. we need to clamp
    if renormalize:
        image_tensor.add_(1).div_(2)

    image_tensor.mul_(255).clamp_(0, 255)

    if len(image_tensor.shape) == 4:
        image_numpy = image_tensor[0].cpu().float().numpy()
    else:
        image_numpy = image_tensor.cpu().float().numpy()

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    return image_numpy.astype(imtype)


def create_grid_from_tensor(tensor_of_images, num_rows=4):
    # take norm over real-imaginary dimension
    # tensor_of_images = tensor_of_images.norm(dim=1, keepdim=True)

    # make image grid
    tensor_grid = tvutil.make_grid(
        tensor_of_images, nrow=num_rows, normalize=True, scale_each=False
    )
    numpy_grid = tensor2im(tensor_grid, renormalize=False)

    return numpy_grid


def gray2heatmap(grayimg, cmap="jet"):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(grayimg)
    # rgb_img = np.delete(rgba_img, 3, 2) * 255.0
    rgb_img = rgba_img[:, :, :, 0] * 255.0
    rgb_img = rgb_img.astype(np.uint8)
    return rgb_img
