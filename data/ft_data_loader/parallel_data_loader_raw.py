
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




def create_mask2(shape, subsampling_ratio, seed=None, subsampling_type='lf_focused'):
    """ Samples a mask that removes rows, one per instance in the batch
        1 indicates the pixel is known during training.
        Expected input shape is: nbatches x 1 x height x width x 2
        or single batch: 1 x height x width x 2.
        The 2 at the end is so this mask can be applied to complex style tensors directly,
        which is pytorch have the 2 complex dimensions at the end.

        Keep in mind that the fft operation in torch places the low frequencies
        at the corners of the tensor, and the high frequencies in the middle.
    """
    original_shape = shape
    if len(original_shape) == 4:
        shape = (1, *original_shape)

    if seed is not None:
        np.random.seed(seed)

    img_mask_np = np.zeros(shape).astype(np.float32) #img_fft_np.copy()
    nbatches = shape[0]
    nrows = shape[2]

    # Just simplfies the implementation
    if not float(subsampling_ratio).is_integer():
        raise Exception("Subsampling ratio should be integral")
    center_fraction = 0.078125
    radius = int(nrows*center_fraction/2)
    # For random methods
    # Do not sample from the radius range
    if subsampling_type == "hf_focused":
        row_pool = np.concatenate([np.arange(0, nrows//2-radius), np.arange(nrows//2+radius, nrows)],0) 
    else:
        # Usually we want to keep lf more
        row_pool = np.arange(radius, nrows-radius) 

    row_sample = np.zeros((nbatches, nrows))
    actual_subsampling_nrow = int((1.0/subsampling_ratio - center_fraction) * nrows) # excluding the radius rows
    # print(row_pool, actual_subsampling_nrow)
    row_ids = np.random.choice(row_pool, size=actual_subsampling_nrow, replace=False) 
    row_sample[:,row_ids] = 1
    sampled_mask = np.expand_dims(row_sample, 1)[..., None, None] # Add extra dimensions
    sampled_mask = np.broadcast_to(sampled_mask, shape) # Expand

    random_row_sample = np.zeros((nbatches, nrows))
    for i in range(nbatches):
        ids = np.random.choice(nrows, size=int(1.0/subsampling_ratio*nrows), replace=False) 
        random_row_sample[i,ids] = 1
    
    # For center masking methods
    middle = int(nrows/2)
    non_center_inds = (
        list(range(radius, middle-radius)) +
        list(range(middle+radius, nrows-radius)))
    
    if subsampling_type == "random":
        for i in range(nbatches):
            # sample per batch
            row_ids = np.random.choice(nrows, size=int((1.0/subsampling_ratio) * nrows), replace=False) 
            img_mask_np[i,:,row_ids,:,:] = 1
    elif subsampling_type == "hf_focused":        
        img_mask_np[:] = sampled_mask
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 1
    elif subsampling_type == "lf_focused":
        # Random sample everywhere
        img_mask_np[:] = sampled_mask
        # Add all low frequencies (Top and bottom)
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
    elif subsampling_type == "lf_focused_no_hf":
        # Random sample everywhere
        img_mask_np[:] = sampled_mask
        # region around either end should be all 1
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
        # region around center of image should be all 0
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 0
    elif subsampling_type == "alternating_plus_hf":
        img_mask_np[...,::ratio,:,:] = 1
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 1
        img_mask_np[...,:radius,:,:] = 0
        img_mask_np[...,(-radius):,:,:] = 0
    elif subsampling_type == "alternating_plus_lf":
        #https://arxiv.org/pdf/1709.02576.pdf
        img_mask_np[...,::ratio,:,:] = 1
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 0
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
    elif subsampling_type == "alternating":
        # Really doesn't work very well!
        img_mask_np[...,::ratio,:,:] = 1
    elif subsampling_type == "fromfile":
        mask2d = load_mask(args.subsample_mask_file)
        img_mask_np_raw = mask2d[None, None, :, :, None]
        img_mask_np =  np.tile(img_mask_np_raw, reps=(nbatches, 1, 1, 2))

    if len(original_shape) == 4:
        img_mask_np = img_mask_np[0, ...]

    return torch.from_numpy(img_mask_np)

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
                # self.mask = create_mask(shape, random_frac=self.random, mask_fraction=self.subsampling_ratio)
                mask = create_mask2([1,shape,shape,2], 1//self.subsampling_ratio) #[1,h,w,2]
                self.mask = mask[:,:,:1,0].unsqueeze(0)
            return self.mask
        else:
            mask = create_mask2([1,shape,shape,2], 1//self.subsampling_ratio) #[1,h,w,2]
            self.mask = mask[:,:,:1,0].unsqueeze(0)
            # self.mask = create_mask(shape, random_frac=self.random, mask_fraction=self.subsampling_ratio)
            # self.mask = create_mask(shape, random_frac=self.random, mask_fraction=self.subsampling_ratio, random_full=True)
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