
import torch
from torch import nn
import numpy as np

# note that for IFFT we do not use irfft
# this function returns two channels where the first one (real part) is in image space
class IFFT(nn.Module):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = torch.ifft(x, 2)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'IFFT()'


# class FFT(nn.Module):
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         y = torch.fft(x, 2)
#         return y.permute(0, 3, 1, 2)

#     def __repr__(self):
#         return 'FFT()'

class RFFT(nn.Module):
    def forward(self, x):
        x = x.squeeze(1)
        y = torch.rfft(x, 2, onesided=False)
        return y.permute(0, 3, 1, 2)

    def __repr__(self):
        return 'RFFT()'

def create_mask(n = 128, mask_fraction = 0.25, mask_low_freqs = 5):
    
    mask_fft = (np.random.RandomState(42).rand(n) < mask_fraction).astype(np.float32)
    mask_fft[:mask_low_freqs] = mask_fft[-mask_low_freqs:] = 1
    mask_fft = torch.from_numpy(mask_fft).view(1, 1, n, 1)
    return mask_fft