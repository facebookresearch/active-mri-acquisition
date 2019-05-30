
import torch
from torch import nn
import numpy as np

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
#     import pdb; pdb.set_trace()
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)
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

# def create_mask(n=128, mask_fraction=0.25, mask_low_freqs=5, seed=42, random_frac=False):
    
#     if random_frac:
#         # we sample fraction and mask_low_freqs lines
#         ratio = np.random.rand(1) + 0.5
#         mask_fraction *= ratio
#         ratio = np.random.rand(1) + 0.5
#         mask_low_freqs = int(mask_low_freqs * ratio)
#         seed = np.random.randint(100)

#     if type(n) is int:
#         b = 1
#         mask_fft = (np.random.RandomState(seed).rand(b,n) < mask_fraction).astype(np.float32)
#     elif type(n) is tuple:
#         assert len(n) == 2
#         b, n = n[0], n[1]
#         mask_fft = (np.random.rand(b,n) < mask_fraction).astype(np.float32)
        

#     mask_fft[:,:mask_low_freqs] = mask_fft[:,-mask_low_freqs:] = 1
#     mask_fft = torch.from_numpy(mask_fft).view(b, 1, n, 1)
    
#     return mask_fft

def create_mask(n=128, mask_fraction=0.25, mask_low_freqs=5, seed=42, 
                random_frac=False, random_full=False, random_lowfreq=False):
    assert random_frac <= 1
    if type(n) is int:
        b = 1
    else:
        b, n = n[0], n[1]
    
    mask_fft = np.zeros((b,n)).astype(np.float32)
    for i in range(b):
        if random_full:
            # random over all rates
            if np.random.rand(1) > 0.7:
                mask_fraction = min(max(0.1, np.random.rand(1)), 0.9)
            else:
                mask_fraction = min(max(0.1, np.random.rand(1)), 0.4)
        elif random_frac:
            # we sample fraction and mask_low_freqs lines
            ratio = np.random.rand(1) + 0.5
            mask_frac = mask_fraction * ratio
            # ratio = np.random.rand(1) + 0.5
            # mask_lf = int(mask_low_freqs * ratio)
            mask_lf = np.random.choice(range(int(mask_low_freqs * 0.5), int(mask_low_freqs * 1.5) + 1))
            seed = np.random.randint(10000)
        elif random_lowfreq:
            expected_num_lines = mask_fraction * (np.random.rand() + 0.5)
            mask_frac = 0
            mask_lf = np.random.binomial(n, expected_num_lines)
        else:
            mask_frac = mask_fraction
            mask_lf = mask_low_freqs
        s_fft = (np.random.RandomState(seed).rand(n) < mask_frac).astype(np.float32)
        mask_fft[i,:] = s_fft
        mask_fft[i,:mask_lf] = mask_fft[i,-mask_lf:] = 1

    mask_fft = torch.from_numpy(mask_fft).view(b, 1, 1, n)

    return mask_fft
