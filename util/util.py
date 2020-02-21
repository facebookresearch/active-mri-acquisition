from __future__ import print_function
import torch
import torchvision.utils as tvutil
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr


def compute_ssims(xs, ys):
    ssims = []
    for i in range(xs.shape[0]):
        ssim = compare_ssim(
            xs[i, 0].cpu().numpy(),
            ys[i, 0].cpu().numpy(),
            win_size=11,
            data_range=ys[i, 0].cpu().numpy().max())
        ssims.append(ssim)
    return np.array(ssims).mean()


def compute_psnrs(xs, ys):
    psnrs = []
    for i in range(xs.shape[0]):
        psnr = compare_psnr(xs[i, 0].cpu().numpy(), ys[i, 0].cpu().numpy(), data_range=ys[i, 0].cpu().numpy().max())
        psnrs.append(psnr)
    return np.array(psnrs).mean()


def ssim_metric(src, tar, full=False, size_average=True):
    return compute_ssims(src, tar)


def psnr_metric(src, tar):
    return compute_psnrs(src, tar)


def sum_axes(input, axes=[], keepdim=False):
    # mu2, logvar2 are prior
    # probably some check for uniqueness of axes
    if axes == -1:
        axes = [i for i in range(1, len(input.shape))]

    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input


def mri_denormalize(input_image, zscore=3):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    # do normalization first, since we working on fourier space. we need to clamp
    for dat in input_image:
        minv = max(-zscore, dat.min())
        maxv = min(zscore, dat.max())
        dat.clamp_(minv, maxv)
        dat.add_(-minv).div_(maxv - minv)

    return input_image


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

    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy.astype(imtype)


def create_grid_from_tensor(tensor_of_images, num_rows=4):
    """

    Args:
        tensor_of_images:   cuda tensor of images to be converted into grid of images
                            shape   :   (batch_size, 2, height, width)

    Returns:

    """
    #take norm over real-imaginary dimension
    # tensor_of_images = tensor_of_images.norm(dim=1, keepdim=True)

    #make image grid
    tensor_grid = tvutil.make_grid(
        tensor_of_images, nrow=num_rows, normalize=True, scale_each=False)
    numpy_grid = tensor2im(tensor_grid, renormalize=False)

    return numpy_grid


def gray2heatmap(grayimg, cmap='jet'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(grayimg)
    # rgb_img = np.delete(rgba_img, 3, 2) * 255.0
    rgb_img = rgba_img[:, :, :, 0] * 255.0
    rgb_img = rgb_img.astype(np.uint8)
    return rgb_img


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (np.mean(x),
                                                                                     np.min(x),
                                                                                     np.max(x),
                                                                                     np.median(x),
                                                                                     np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
