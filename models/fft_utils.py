import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# note that for IFFT we do not use irfft
# this function returns two channels where the first one (real part) is in image space
class IFFT(nn.Module):

    def forward(self, x, normalized=False):
        x = x.permute(0, 2, 3, 1)
        y = torch.ifft(x, 2, normalized=normalized)
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


def clamp(tensor):
    # TODO: supposed to be clamping to zscore 3, make option for this
    return tensor.clamp(-3, 3)


def gaussian_nll_loss(reconstruction, target, logvar):
    if reconstruction.shape[2] == 640:
        reconstruction = reconstruction[:, :, 160:-160, 24:-24]
        target = target[:, :, 160:-160, 24:-24]
        logvar = logvar[:, :, 160:-160, 24:-24]
    l2 = F.mse_loss(reconstruction[:, :1, :, :], target[:, :1, :, :], reduce=False)
    # Clip logvar to make variance in [0.01, 5], for numerical stability
    logvar = logvar.clamp(-4.605, 1.609)
    one_over_var = torch.exp(-logvar)

    assert len(l2) == len(logvar)
    return 0.5 * (one_over_var * l2 + logvar)


# TODO fix the conditional return
def preprocess_inputs(target,
                      mask,
                      fft_functions,
                      options,
                      return_masked_k_space=False,
                      clamp_target=True):
    if clamp_target:
        target = clamp(target.to(options.device)).detach()

    if hasattr(options, 'dynamic_mask_type') and options.dynamic_mask_type != 'loader':
        mask = create_mask(
            target.shape[0], num_entries=mask.shape[3], mask_type=options.dynamic_mask_type,
            low_freq_count=options.low_freq_count)
    mask = mask.to(options.device)

    if options.dataroot == 'KNEE_RAW':
        target = torch.norm(
            target, p=2, dim=3, keepdim=True)  #TODO: to be updated based on decision
        target = target.permute(0, 3, 1, 2)
    masked_true_k_space = fft_functions['rfft'](target) * mask
    zero_filled_reconstruction = fft_functions['ifft'](masked_true_k_space)

    target = torch.cat([target, torch.zeros_like(target)], dim=1)

    if return_masked_k_space:
        return zero_filled_reconstruction, target, mask, masked_true_k_space
    return zero_filled_reconstruction, target, mask


def create_mask(batch_size, num_entries=128, mask_type='random', low_freq_count=5):
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
