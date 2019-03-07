# Transforms for the MRI dataloaders. Inspired by Matthew Muckley's code
# (https://github.com/nyu-med-ai/knet-recon/tree/master/mriutils)

import unittest

import torch
import numpy as np
from numpy.fft import ifftshift, fftshift, ifft2
from torchvision.transforms import Compose


class ToTensor:
    """Convert numpy array to PyTorch.

    Can handle complex numpy arrays by stacking the real and imaginary
    components in the last dimension."""

    def __call__(self, sample):
        """
        Args:
            sample (dictionary): sample['data'] is the input

        Returns:
            dictionary: sample['data'] converted to PyTorch Tensor
        """
        if np.iscomplexobj(sample['data']):
            x = np.stack((sample['data'].real, sample['data'].imag), axis=-1)
            sample['data'] = torch.from_numpy(x)
        else:
            sample['data'] = torch.from_numpy(sample['data'])
        return sample


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


class Mask:
    """Multiply with mask."""

    def __init__(self, mask_func):
        """
        Args:
            mask_func (callable): A function that takes a shape (tuple of ints)
            and a seed (tuple of ints) and returns a mask (Tensor).
        """
        self.mask_func = mask_func

    def __call__(self, sample):
        """
        Args:
            sample (dictionary): sample['data'] is the input tensor,
            sample['seed'] is the seed that's passed to the mask function.

        Returns:
            dictionary: sample['data'] is masked; mask is stored in
            sample['mask'].
        """
        mask = self.mask_func(sample['data'].shape, sample['seed'])
        sample['mask'] = mask
        sample['data'] *= mask
        return sample


def apply_mask(data, mask_func, args):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, args)
    if isinstance(data, torch.Tensor):
        mask = to_tensor(mask)

    return data * mask, mask


def rfft2(x):
    return torch.rfft(x, signal_ndim=2, onesided=False, normalized=True)


def irfft2(x):
    return torch.irfft(x, signal_ndim=2, onesided=False, normalized=True)


def cfft2(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.fft2(x, norm='ortho')
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        assert x.size(-3) % 2 == 0
        assert x.size(-2) % 2 == 0
        assert x.size(-1) == 2
        x = fftshift(x, dim=(-3, -2))
        x = torch.fft(x, 2, normalized=True)
        x = fftshift(x, dim=(-3, -2))
        return x
    assert False


def cifft2(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.ifft2(x, norm='ortho')
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        assert x.size(-3) % 2 == 0
        assert x.size(-2) % 2 == 0
        assert x.size(-1) == 2
        x = ifftshift(x, dim=(-3, -2))
        x = torch.ifft(x, 2, normalized=True)
        x = fftshift(x, dim=(-3, -2))
        return x
    assert False


class Abs:
    """Absolute value of a complex Tensor (size of last dimension == 2)."""

    def __call__(self, sample):
        """
        Args:
            sample (dictionary): sample['data'] is the input Tensor.

        Returns:
            dictionary: sample['data'] contains the absolute value of the
            input.
        """
        assert sample['data'].size(-1) == 2
        sample['data'] = (sample['data']**2).sum(dim=-1).sqrt()
        return sample


def complex_abs(data, eps=1e-20):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return ((data ** 2).sum(dim=-1) + eps).sqrt()


def center_crop(x, shape):
    assert 0 < shape[0] <= x.shape[-2]
    assert 0 < shape[1] <= x.shape[-1]
    w_from = (x.shape[-1] - shape[0]) // 2
    h_from = (x.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[..., h_from:h_to, w_from:w_to]
    return x


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


class Normalize:
    """Subtract the mean and divide by the standard deviation"""

    def __init__(self, mean, std):
        """
        Args:
            mean (float): The mean.
            std (float): The standard deviation.
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            sample (dictionary): sample['data'] is the input Tensor.

        Returns:
            dictionary: sample['data'] is the normalized input.
        """
        sample['data'] -= self.mean
        sample['data'] /= self.std
        return sample


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions
# Not sure if the following functions belong in this file since they might be
# useful elsewhere. Maybe move them to a new file common/utils.py?

def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def roll(x, shift, dim):
    """Same as np.roll"""
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """Same as np.fft.fftshift"""
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


class Test(unittest.TestCase):
    def test_roll(self):
        args = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (100, 0),
            (1, 1),
            (2, 2),
            ((1, 2), (0, 1)),
            ((1, 2), (0, 2)),
            ((1, 2), (1, 2)),
        ]
        a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        for shift, dim in args:
            out_th = roll(torch.from_numpy(a), shift, dim).numpy()
            out_np = np.roll(a, shift, dim)
            np.testing.assert_allclose(out_th, out_np)

    def test_fftshift(self):
        a = np.arange(2 * 4 * 6).reshape(2, 4, 6)
        out_th = fftshift(torch.from_numpy(a)).numpy()
        out_np = np.fft.fftshift(a)
