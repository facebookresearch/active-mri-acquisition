
import os

import numpy as np
import torch
from matplotlib.image import imsave
from torch.utils import data
from torch.utils.data import DataLoader
import sys
# sys.path.insert(0,'../../../')
# from common import subsample
# from common.args import Args
# from common.dicom_dataset import Slice
# from mri.model import ifft
# from multicoil_compression import MriCoilCompr
from torchvision import utils
from models.fft_utils import create_mask
import time
import math
import numpy as np
import scipy
import torch
from scipy.ndimage import gaussian_filter
from scipy import sparse
from scipy.sparse.linalg import svds



class MriCoilCompr(object):
    """ Apply coil compression using principal components analysis.
        This used for learning to present the network with a somewhat
        standardized coil architecture and coil count.

    """

    def __init__(self, out_coils=8, verbose=False):
        self.out_coils = out_coils
        self.verbose = verbose

    def __call__(self, dat, u=None):
        
        """
        Args:
            dat (array):  [ncoil, height, width, slices]
        Returns:
            dat (array):  [ncoil, height, width, slices]
        """
        start_time = time.time()
        # target, dat = sample['target'], sample['dat']

        datdims = dat.shape
        dat = np.reshape(dat, (datdims[0], np.prod(datdims[1:])))
        dat = np.matrix(dat)
        raw_u = None
        if u is None:
            if (self.out_coils < datdims[0]-1):
                u = np.matrix(scipy.sparse.linalg.svds(dat, 
                    k=self.out_coils,
                    return_singular_vectors='u')[0])
                n_coils = self.out_coils
            else:
                u, s, vh = np.linalg.svd(dat, full_matrices=False)
                n_coils = min(u.shape[1], self.out_coils)
            raw_u = u

        dat = np.reshape(np.asarray(u[:,:self.out_coils].H * dat), 
                (n_coils,) + datdims[1:])
        dat = np.concatenate((dat, 
                np.zeros((self.out_coils-n_coils,) + datdims[1:])))
        
        end_time = time.time()
          
        if self.verbose:
            print('coil compr time: %f' % (end_time-start_time))

        if raw_u is not None:
            return dat, raw_u
        else:
            return dat

    def __repr__(self):
        return self.__class__.__name__ + '(snr_range)'.format(self.snr_range)

shapes = {
    'train': (40746, 320, 320, 1, 15),
    'val': (5081, 320, 320, 1, 15),
    'public_leaderboard': (5692, 320, 320, 1, 15),
    'private_leaderboard': (6323, 320, 320, 1, 15),
}

class Mask:
    def __init__(self, reuse_mask, subsampling_ratio, random):
        self.mask = None
        self.reuse_mask = reuse_mask
        self.subsampling_ratio = subsampling_ratio
        self.random = random
        print(f'[Mask] reuse_mask = {reuse_mask} subsampling ratio = {subsampling_ratio} random {random}')

    def __call__(self, shape):
        if self.reuse_mask:
            if self.mask is None:
                self.mask = create_mask(shape, random_frac=self.random, mask_fraction=self.subsampling_ratio)
            return self.mask
        else:
            self.mask = create_mask(shape, random_frac=self.random, mask_fraction=self.subsampling_ratio)
            return self.mask

class PCASingleCoilSlice(data.Dataset):
    def __init__(self, mask_func, root, which='train', use_clip=True):
        self.mask_func = mask_func
        self.root = root
        self.images = np.memmap(os.path.join(root, which, 'mmap_data.dat'), mode='r', dtype=np.float32, shape=shapes[which])
        self.means = np.load(os.path.join(root, which, 'means.npy'))
        self.stds = np.load(os.path.join(root, which, 'stds.npy'))
        self.pcac = MriCoilCompr(1)
        self.use_clip = use_clip
        self.zscore = 10
        if use_clip:
            print (f'[PCASingleCoilSlice] -> use zscore {self.zscore} clip internally')

    def __getitem__(self, index):
        slice = self.images[index]
        dim_x, dim_y, _, num_coils = slice.shape

        mean = self.means[index]
        std = self.stds[index]
        volume_metadata = {'mean': mean, 'std': std}
        slice = np.transpose(slice[:,:,0,:], axes=(2,0,1))

        sc_slice,_ = self.pcac(slice) # (1, 320, 320)
        # have to normalize after PCA
        sc_slice = (sc_slice - mean) / std
        
        mask = self.mask_func(dim_x)

        target_image = torch.from_numpy(np.concatenate([sc_slice.real, sc_slice.imag],axis=0).astype(np.float32)) # (2, 320, 320)
        if self.use_clip:
            target_image.clamp_(-self.zscore, self.zscore)
        slice_fft = torch.fft(target_image.permute(1,2,0).unsqueeze(0), 2, normalized=True)
        slice_fft = slice_fft.permute(0, 3, 1, 2) * mask

        slice_ifft_img = torch.ifft(slice_fft.permute(0,2,3,1), 2, normalized=True)
        slice_ifft_img = slice_ifft_img.permute(0, 3, 1, 2).squeeze(0)
        mask = mask.squeeze(0)

        return slice_ifft_img, target_image, mask, volume_metadata

    def __len__(self):
        return self.images.shape[0]

if __name__ == '__main__':
    root = '/private/home/zizhao/work/mri_data/multicoil/raw_mmap/FBAI_Knee/'
    shape = (57842, 320, 320, 1, 15)
    args = Args().parse_args()
    mask_func = Mask(reuse_mask=False)

    data = PCASingleCoilSlice(mask_func, args, root)
    # data = Slice(mask_func, args, resolution=320)
    train_loader = DataLoader(
        dataset=data,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    os.system('rm loader_res/*.png')
    for j, (slice_fft, target_image, mask, volume_metadata) in enumerate(train_loader):
        print(slice_fft.shape, target_image.shape, mask.shape)
        target_image.clamp_(-10,10) # some intensities are too high that suppress overall intensities too be soo low
        vis = target_image.norm(2,dim=1,keepdim=True)
        utils.save_image(vis, f'loader_res/img_{j}.png', normalize=True, scale_each=True)
        vis = slice_fft.norm(2,dim=1,keepdim=True)
        utils.save_image(vis, f'loader_res/img_{j}_masked.png', normalize=True, scale_each=True)
        mask = mask.repeat(1,1,1,320)
        utils.save_image(mask, f'loader_res/mask_{j}.png', normalize=True, scale_each=True)