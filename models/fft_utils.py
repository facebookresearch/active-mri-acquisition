
import torch
from torch import nn
import numpy as np


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


def create_mask(batch_size, num_entries=128, mask_type='random'):
    mask = np.zeros((batch_size, num_entries)).astype(np.float32)
    for i in range(batch_size):
        if mask_type == 'random_zz':
            mask_fraction = 0.25
            mask_low_freqs = 5
            # we sample fraction and mask_low_freqs lines
            ratio = np.random.rand(1) + 0.5
            mask_frac = mask_fraction * ratio
            mask_lf = np.random.choice(range(int(mask_low_freqs * 0.5), int(mask_low_freqs * 1.5) + 1))
            seed = np.random.randint(10000)
            s_fft = (np.random.RandomState(seed).rand(num_entries) < mask_frac).astype(np.float32)
            mask[i, :] = s_fft
            mask[i, :mask_lf] = mask[i,-mask_lf:] = 1
        elif mask_type == 'random_lowfreq':
            p_lines = 0.25 * np.random.random() + 0.125     # ~U(0.125, 0/375)
            num_low_freq_lines = np.random.binomial(num_entries, p_lines)
            mask[i, :num_low_freq_lines] = mask[i, -num_low_freq_lines:] = 1
        else:
            raise ValueError('Invalid mask type: {}.'.format(mask_type))

    return torch.from_numpy(mask).view(batch_size, 1, 1, num_entries)
