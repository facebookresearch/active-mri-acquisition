import numpy as np
import pdb
import torch
from torch.nn.functional import avg_pool2d
from . import common_utils

def save_row_mask(mask, fname):
    """
        Save a 1D numpy binary array as a row mask png image
    """
    mask2d = np.tile(mask, reps=(len(mask1), 1)).T # Tile it to 2D
    return save_mask(mask2d, fname)

def save_mask(mask, fname):
    """
    Save a 2D numpy binay array out as a png image
    """
    common_utils.save_np_img(mask, fname)

def load_mask(fname):
    """
        Returns the png file mask as a 2D numpy array
    """
    img_pil, _ = common_utils.get_image(fname, -1)
    mask = common_utils.pil_to_np(img_pil) # 1xnxn
    return mask[0, :, :]

def build_mask(shape, args, seed=None):
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

    ratio = args.subsampling_ratio
    img_mask_np = np.zeros(shape).astype(np.float32) #img_fft_np.copy()
    nbatches = shape[0]
    nrows = shape[2]

    # Just simplfies the implementation
    if not float(ratio).is_integer():
        raise Exception("Subsampling ratio should be integral")
    
    radius = int(nrows*args.center_fraction/2)
    
    # For random methods
    # Do not sample from the radius range
    if args.subsampling_type == "hf_focused":
        row_pool = np.concatenate([np.arange(0, nrows//2-radius), np.arange(nrows//2+radius, nrows)],0) 
    else:
        # Usually we want to keep lf more
        row_pool = np.arange(radius, nrows-radius) 

    row_sample = np.zeros((nbatches, nrows))
    actual_subsampling_nrow = int((1.0/args.subsampling_ratio - args.center_fraction) * nrows) # excluding the radius rows
    row_ids = np.random.choice(row_pool, size=actual_subsampling_nrow, replace=False) 
    row_sample[:,row_ids] = 1
    sampled_mask = np.expand_dims(row_sample, 1)[..., None, None] # Add extra dimensions
    sampled_mask = np.broadcast_to(sampled_mask, shape) # Expand

    random_row_sample = np.zeros((nbatches, nrows))
    for i in range(nbatches):
        ids = np.random.choice(nrows, size=int(1.0/args.subsampling_ratio*nrows), replace=False) 
        random_row_sample[i,ids] = 1
    
    # For center masking methods
    middle = int(nrows/2)
    non_center_inds = (
        list(range(radius, middle-radius)) +
        list(range(middle+radius, nrows-radius)))
    
    if args.subsampling_type == "random":
        for i in range(nbatches):
            # sample per batch
            row_ids = np.random.choice(nrows, size=int((1.0/args.subsampling_ratio) * nrows), replace=False) 
            img_mask_np[i,:,row_ids,:,:] = 1
    elif args.subsampling_type == "hf_focused":        
        img_mask_np[:] = sampled_mask
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 1
    elif args.subsampling_type == "lf_focused":
        # Random sample everywhere
        img_mask_np[:] = sampled_mask
        # Add all low frequencies (Top and bottom)
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
    elif args.subsampling_type == "lf_focused_no_hf":
        # Random sample everywhere
        img_mask_np[:] = sampled_mask
        # region around either end should be all 1
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
        # region around center of image should be all 0
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 0
    elif args.subsampling_type == "alternating_plus_hf":
        img_mask_np[...,::ratio,:,:] = 1
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 1
        img_mask_np[...,:radius,:,:] = 0
        img_mask_np[...,(-radius):,:,:] = 0
    elif args.subsampling_type == "alternating_plus_lf":
        #https://arxiv.org/pdf/1709.02576.pdf
        img_mask_np[...,::ratio,:,:] = 1
        img_mask_np[...,(middle-radius):(middle+radius),:,:] = 0
        img_mask_np[...,:radius,:,:] = 1
        img_mask_np[...,(-radius):,:,:] = 1
    elif args.subsampling_type == "alternating":
        # Really doesn't work very well!
        img_mask_np[...,::ratio,:,:] = 1
    elif args.subsampling_type == "fromfile":
        mask2d = load_mask(args.subsample_mask_file)
        img_mask_np_raw = mask2d[None, None, :, :, None]
        img_mask_np =  np.tile(img_mask_np_raw, reps=(nbatches, 1, 1, 2))

    if len(original_shape) == 4:
        img_mask_np = img_mask_np[0, ...]

    return img_mask_np


def downsample_from_fft(fft_data, factor):
    """
        Takes a pytorch minibatch, with shape nbatchx1xheightxwidthx2
        and produces a ifft of the low frequencies only, giving
        an image of height/factor, width/factor size
    """
    if fft_data.shape[2] != fft_data.shape[3]:
        raise Exception("Image not square")
    n = fft_data.shape[2]
    if factor < 1 and isinstance(factor, int):
        raise Exception(f"need factor {factor} to be integer and >= 1")

    radius = n//(2*factor)

    mri_fft_mask = torch.zeros_like(fft_data)
    mri_fft_mask[:,:,:radius,:radius,:] = 1.0
    mri_fft_mask[:,:,:radius,(n-radius):,:] = 1.0
    mri_fft_mask[:,:,(n-radius):,:radius,:] = 1.0
    mri_fft_mask[:,:,(n-radius):,(n-radius):,:] = 1.0
    mri_fft_rescale = fft_data * mri_fft_mask

    rescaled_recon = torch.ifft(mri_fft_rescale, signal_ndim=2).transpose(1, 4)[..., 0]

    # TODO support non-integer factors
    rescaled_downsampled = avg_pool2d(rescaled_recon,
        kernel_size=(factor,factor), stride=factor, padding=0)

    return rescaled_downsampled

class Mask:
    def __init__(self, reuse_mask):
        self.mask = None
        self.reuse_mask = reuse_mask

    def __call__(self, shape, args):
        if self.reuse_mask:
            if self.mask is None:
                self.mask = build_mask(shape, args, args.subsampling_seed)
            return self.mask
        else:
            return build_mask(shape, args)
