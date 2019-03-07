"""Tensors with complex-valued entries.
"""

import ctypes
import functools

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

tensor_base = torch.Tensor().__class__.__bases__[0]


class ComplexTensor:

    def __init__(self, real, imag):
        assert real.size() == imag.size()
        self.real = real
        self.imag = imag

    def __add__(self, X):
        return ComplexTensor(self.real + X.real, self.imag + X.imag)

    def __sub__(self, X):
        return ComplexTensor(self.real - X.real, self.imag - X.imag)

    def __mul__(self, a):
        if isinstance(a, (int, float)):
            return ComplexTensor(self.real * a, self.imag * a)
        elif isinstance(a, tensor_base):
            return ComplexTensor(self.real * a, self.imag * a)
        elif isinstance(a, ComplexTensor):
            real = self.real * a.real - self.imag * a.imag
            imag = self.real * a.imag + self.imag * a.real
            return ComplexTensor(real, imag)
        else:
            raise NotImplementedError

    def __truediv__(self, a):
        assert isinstance(a, (int, float))
        return ComplexTensor(self.real / a, self.imag / a)

    def __rtruediv__(self, a):
        assert isinstance(a, (int, float))
        return ComplexTensor(a / self.real, a / self.imag)

    def __matmul__(self, X):
        real = self.real @ X.real - self.imag @ X.imag
        imag = self.real @ X.imag + self.imag @ X.real
        return ComplexTensor(real, imag)

    def __abs__(self):
        return torch.sqrt(self.real**2 + self.imag**2)

    def __setitem__(self, key, item):
        self.real[key] = item.real
        self.imag[key] = item.imag

    def __getitem__(self, key):
        return ComplexTensor(self.real[key], self.imag[key])

    def t(self):
        return ComplexTensor(self.real.t(), self.imag.t())

    def h(self):
        return ComplexTensor(self.real.t(), -self.imag.t())

    def conj(self):
        return ComplexTensor(self.real, -self.imag)

    def transpose(self, dim0, dim1):
        return ComplexTensor(self.real.transpose(dim0, dim1),
                             self.imag.transpose(dim0, dim1))

    def permute(self, *args):
        return ComplexTensor(self.real.permute(*args),
                             self.imag.permute(*args))

    def contiguous(self):
        return ComplexTensor(self.real.contiguous(), self.imag.contiguous())

    @property
    def shape(self):
        return self.real.shape

    def size(self):
        return self.real.shape

    def numel(self):
        return self.real.numel()

    def numpy(self):
        return self.real.cpu().numpy() + 1j * self.imag.cpu().numpy()

    def new(self, *args):
        return ComplexTensor(self.real.new(*args), self.imag.new(*args))

    def view(self, *args):
        return ComplexTensor(self.real.view(*args), self.imag.view(*args))

    def unsqueeze(self, *args):
        return ComplexTensor(self.real.unsqueeze(*args),
                             self.imag.unsqueeze(*args))

    def squeeze(self, *args):
        return ComplexTensor(self.real.squeeze(*args),
                             self.imag.squeeze(*args))

    def repeat(self, *args):
        return ComplexTensor(self.real.repeat(*args), self.imag.repeat(*args))

    def zero_(self):
        self.real.zero_()
        self.imag.zero_()
        return self

    def normal_(self, *args, **kwargs):
        self.real.normal_(*args, **kwargs)
        self.imag.normal_(*args, **kwargs)
        return self

    def double(self):
        return ComplexTensor(self.real.double(), self.imag.double())

    def float(self):
        return ComplexTensor(self.real.float(), self.imag.float())

    def cuda(self):
        return ComplexTensor(self.real.cuda(), self.imag.cuda())

    def cpu(self):
        return ComplexTensor(self.real.cpu(), self.imag.cpu())

    def requires_grad_(self):
        self.real.requires_grad_()
        self.imag.requires_grad_()
        return self

    @property
    def data(self):
        return ComplexTensor(self.real.data, self.imag.data)

    @property
    def variable(self):
        return ComplexTensor(Variable(self.real), Variable(self.imag))

    def __repr__(self):
        return 'real{}\nimag{}'.format(repr(self.real), repr(self.imag))


def bmm(x, y):
    real = torch.bmm(x.real, y.real) - torch.bmm(x.imag, y.imag)
    imag = torch.bmm(x.real, y.imag) + torch.bmm(x.imag, y.real)
    return ComplexTensor(real, imag)


def cat(seq, **kwargs):
    """Concatenates together the terms in the given sequence.
    """
    real = torch.cat([x.real for x in seq], **kwargs)
    imag = torch.cat([x.imag for x in seq], **kwargs)
    return ComplexTensor(real, imag)


def norm(X, **kwargs):
    """torch.norm after taking absolute values entrywise.
    """
    return torch.norm(abs(X), **kwargs)


def from_numpy(x):
    """Creates a ComplexTensor from a numpy.ndarray.
    """
    return ComplexTensor(torch.from_numpy(x.real), torch.from_numpy(x.imag))


# FFT
#
# Wrapper for the cuFFT library that mimics the `numpy.fft` API. It is designed
# to work well for cases where `fft` is called multiple times on tensors with
# the same sizes.
#
# Creating a new FFT plan is an expensive operation. This wrapper caches FFT
# plans and reuses them whenever it's called on a tensors whose size it has
# encountered before. The library we were using before created a new plan each
# time `fft` was called.
#
# The wrapper does not currently destroy the FFT plans so it's possible to run
# into memory problems if calling `fft` with different input sizes many, many
# times. If this ever becomes an issue it shouldn't be too hard to modify the
# wraper to delete old plans.

cufft = ctypes.cdll.LoadLibrary('libcufft.so')
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1
CUFFT_SUCCESS = 0x0
CUFFT_Z2Z = 0x69


@functools.lru_cache(maxsize=None)
def _plan(*shape):
    batch_size, *dims = shape
    plan = ctypes.c_int()
    n = (ctypes.c_int * len(dims))(*dims)
    assert cufft.cufftPlanMany(
        ctypes.byref(plan), len(dims), ctypes.byref(n), None, 0, 0, None, 0, 0,
        CUFFT_Z2Z, batch_size) == CUFFT_SUCCESS
    return plan


def _fft(x, rank, direction, plan_cache={}):
    assert isinstance(x, ComplexTensor)
    assert isinstance(x.real, torch.cuda.DoubleTensor)
    assert x.real.dim() >= rank
    orig_shape = x.shape
    if x.real.dim() == rank:
        x = x.unsqueeze(0)
    else:
        x = x.view(-1, *x.shape[-rank:])
    plan = _plan(*x.shape)
    x_stack = torch.stack((x.real, x.imag), dim=rank + 1)
    y_stack = x_stack.new(x_stack.shape)
    assert cufft.cufftExecZ2Z(
        plan, ctypes.c_void_p(x_stack.data_ptr()),
        ctypes.c_void_p(y_stack.data_ptr()), direction) == CUFFT_SUCCESS
    torch.cuda.synchronize()
    y_real, y_imag = y_stack.split(1, dim=rank + 1)
    y = ComplexTensor(y_real, y_imag).contiguous().view(orig_shape)
    if direction == CUFFT_INVERSE:
        y = y / float(np.prod(x.shape[-rank:]))
    return y


def fft(x):
    return _fft(x, 1, CUFFT_FORWARD)


def ifft(x):
    return _fft(x, 1, CUFFT_INVERSE)


def fft2(x):
    return _fft(x, 2, CUFFT_FORWARD)


def ifft2(x):
    return _fft(x, 2, CUFFT_INVERSE)


# nn
def cwrap(module):
    class CWrap(nn.Module):
        def __init__(self, *args, bias=True, **kwargs):
            super().__init__()
            self.real = module(*args, bias=bias, **kwargs)
            self.imag = module(*args, bias=False, **kwargs)

        def forward(self, x):
            real = self.real(x.real) - self.imag(x.imag)
            imag = self.real(x.imag) + self.imag(x.real)
            return ComplexTensor(real, imag)

        def __repr__(self):
            return 'c' + repr(self.real)
    return CWrap


Linear = cwrap(nn.Linear)
Conv2d = cwrap(nn.Conv2d)


class ReLU(nn.ReLU):
    def forward(self, x):
        return ComplexTensor(super().forward(x.real), super().forward(x.imag))

    def __repr__(self):
        return 'c' + super().__repr__()


# functional
def mse_loss(output, target, **kwargs):
    real = F.mse_loss(output.real, target.real, **kwargs)
    imag = F.mse_loss(output.imag, target.imag, **kwargs)
    return real + imag
