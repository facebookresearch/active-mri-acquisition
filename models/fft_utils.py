import warnings

import numpy as np
import torch

# note that for IFFT we do not use irfft
# this function returns two channels where the first one (real part) is in image space
from torch import nn as nn
from torch.nn import functional as F
from data.ft_data_loader.ft_util_vaes import ifftshift


class IFFT(nn.Module):

    def forward(self, x, normalized=False, ifft_shift=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.ifft(x, 2, normalized=normalized)
        if ifft_shift:
            y = ifftshift(y, dim=(1, 2))
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'IFFT()'


class IRFFT(nn.Module):

    def forward(self, x, normalized=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.irfft(x, 2, onesided=False, normalized=normalized).unsqueeze(3)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'IRFFT()'


class RFFT(nn.Module):

    def forward(self, x, normalized=False):
        # x is in gray scale and has 1-d in the 1st dimension
        x = x.squeeze(1)
        y = torch.rfft(x, 2, onesided=False, normalized=normalized)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'RFFT()'


class FFT(nn.Module):

    def forward(self, x, normalized=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.fft(x, 2, normalized=normalized)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'FFT()'


def center_crop(x, shape):
    assert 0 < shape[0] <= x.shape[-2]
    assert 0 < shape[1] <= x.shape[-1]
    w_from = (x.shape[-1] - shape[0]) // 2
    h_from = (x.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    x = x[..., h_from:h_to, w_from:w_to]
    return x


def to_magnitude(tensor, also_clamp_and_scale=False):
    if also_clamp_and_scale:
        tensor = clamp_and_scale(tensor)
    tensor = (tensor[:, 0, :, :]**2 + tensor[:, 1, :, :]**2)**.5
    return tensor.unsqueeze(1)


def clamp_and_scale(tensor):
    # TODO: supposed to be clamping to zscore 3, make option for this
    return clamp(tensor) + 3


def clamp(tensor):
    # TODO: supposed to be clamping to zscore 3, make option for this
    return tensor.clamp(-3, 3)


def gaussian_nll_loss(reconstruction, target, logvar, options):
    reconstruction = to_magnitude(
        reconstruction, also_clamp_and_scale=options.dataroot != 'KNEE_RAW')
    target = to_magnitude(target, also_clamp_and_scale=options.dataroot != 'KNEE_RAW')
    if options.dataroot == 'KNEE_RAW':
        reconstruction = center_crop(reconstruction, [320, 320])
        target = center_crop(target, [320, 320])
        logvar = center_crop(logvar, [320, 320])
    l2 = F.mse_loss(reconstruction, target, reduce=False)
    # Clip logvar to make variance in [0.01, 5], for numerical stability
    logvar = logvar.clamp(-4.605, 1.609)
    one_over_var = torch.exp(-logvar)

    assert len(l2) == len(logvar)
    return 0.5 * (one_over_var * l2 + logvar)


# TODO fix the conditional return
def preprocess_inputs(batch, fft_functions, options, return_masked_k_space=False,
                      clamp_target=True):

    if options.dataroot == 'KNEE_RAW':
        mask = batch[0].to(options.device)
        # mask = torch.from_numpy(np.fft.ifftshift(mask.numpy(), axes=3)).to(options.device)
        target = batch[1].to(options.device)
        kspace = batch[2].permute(0, 3, 1, 2).to(options.device)
        masked_true_k_space = torch.where(mask.byte(), kspace, torch.tensor(0.).to(options.device))
        zero_filled_reconstruction = fft_functions['ifft'](masked_true_k_space, ifft_shift=True)
        target = target.permute(0, 3, 1, 2)
    else:
        target = batch[1].to(options.device)
        if clamp_target:
            target = clamp(target)
        mask = batch[0].to(options.device)
        fft_target = fft_functions['rfft'](target)
        masked_true_k_space = torch.where(mask.byte(), fft_target,
                                          torch.tensor(0.).to(options.device))
        zero_filled_reconstruction = fft_functions['ifft'](masked_true_k_space)
        target = torch.cat([target, torch.zeros_like(target)], dim=1)
    if return_masked_k_space:
        return zero_filled_reconstruction, target, mask, masked_true_k_space
    return zero_filled_reconstruction, target, mask


def create_mask(batch_size, num_entries=128, mask_type='random', low_freq_count=5):
    warnings.warn(
        'This function has been replaced by the code in data.ft_data_loader.ft_util_vaes.py',
        DeprecationWarning)

    mask = np.zeros((batch_size, num_entries)).astype(np.float32)
    for i in range(batch_size):
        if mask_type == 'random_zz':
            mask_fraction = 0.25
            mask_low_freqs = low_freq_count
            # we sample fraction and mask_low_freqs lines
            ratio = np.random.rand(1) + 0.5
            mask_frac = mask_fraction * ratio
            mask_lf = np.random.choice(
                range(int(mask_low_freqs * 0.5),
                      int(mask_low_freqs * 1.5) + 1))
            seed = np.random.randint(10000)
            s_fft = (np.random.RandomState(seed).rand(num_entries) < mask_frac).astype(np.float32)
            mask[i, :] = s_fft
            mask[i, :mask_lf] = mask[i, -mask_lf:] = 1
        elif mask_type == 'random_lowfreq':
            p_lines = 0.25 * np.random.random() + 0.125  # ~U(0.125, 0.375)
            num_low_freq_lines = np.random.binomial(num_entries, p_lines)
            mask[i, :num_low_freq_lines] = mask[i, -num_low_freq_lines:] = 1
        elif mask_type == 'beta_symmetric':
            half_entries = num_entries // 2
            p = np.random.beta(1, 4)
            mask_lf = np.random.choice(range(5, 11))
            s_fft = (np.random.random(half_entries) < p).astype(np.float32)
            mask[i, :half_entries] = s_fft
            mask[i, :mask_lf] = 1
            mask[i, :-(half_entries + 1):-1] = mask[i, :half_entries]
        else:
            raise ValueError('Invalid mask type: {}.'.format(mask_type))

    return torch.from_numpy(mask).view(batch_size, 1, 1, num_entries)


class GANLossKspace(nn.Module):

    def __init__(self, use_lsgan=True, use_mse_as_energy=False, grad_ctx=False, gamma=100):
        super(GANLossKspace, self).__init__()
        # self.register_buffer('real_label', torch.ones(imSize, imSize))
        # self.register_buffer('fake_label', torch.zeros(imSize, imSize))
        self.grad_ctx = grad_ctx
        if use_lsgan:
            self.loss = nn.MSELoss(size_average=False)
        else:
            self.loss = nn.BCELoss(size_average=False)
        self.use_mse_as_energy = use_mse_as_energy
        if use_mse_as_energy:
            self.FFT = FFT()
            self.gamma = gamma
            self.bin = 5

    def get_target_tensor(self, input, target_is_real, degree, mask, pred_and_gt=None):

        if target_is_real:
            target_tensor = torch.ones_like(input)
            target_tensor[:] = degree

        else:
            target_tensor = torch.zeros_like(input)
            if not self.use_mse_as_energy:
                if degree != 1:
                    target_tensor[:] = degree
            else:
                pred, gt = pred_and_gt
                w = gt.shape[2]
                ks_gt = self.FFT(gt, normalized=True)
                ks_input = self.FFT(pred, normalized=True)
                ks_row_mse = F.mse_loss(
                    ks_input, ks_gt, reduce=False).sum(
                        1, keepdim=True).sum(
                            2, keepdim=True).squeeze() / (2 * w)
                energy = torch.exp(-ks_row_mse * self.gamma)

                # do some bin process
                # import pdb; pdb.set_trace()
                # energy = torch.floor(energy * 10 / self.bin) * self.bin / 10

                target_tensor[:] = energy
            # force observed part to always
            for i in range(mask.shape[0]):
                idx = torch.nonzero(mask[i, 0, 0, :])
                target_tensor[i, idx] = 1
        return target_tensor

    def __call__(self, input, target_is_real, mask, degree=1, updateG=False, pred_and_gt=None):
        # input [B, imSize]
        # degree is the realistic degree of output
        # set updateG to True when training G.
        target_tensor = self.get_target_tensor(input, target_is_real, degree, mask, pred_and_gt)
        b, w = target_tensor.shape
        if updateG and not self.grad_ctx:
            mask_ = mask.squeeze()
            # maskout the observed part loss
            masked_input = torch.where((1 - mask_).byte(), input, torch.tensor(0.).to(input.device))
            masked_target = torch.where((1 - mask_).byte(), target_tensor,
                                        torch.tensor(0.).to(input.device))
            return self.loss(masked_input, masked_target) / (1 - mask_).sum()
        else:
            return self.loss(input, target_tensor) / (b * w)
