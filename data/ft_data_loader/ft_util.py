import numpy as np
# from numpy.fft import fftshift, ifftshift, fftn, ifftn
import torch


def kspace_to_image(kspace_tensor):
    # return (1,H,W)   
    inv_img = torch.irfft(kspace_tensor, 2, onesided=False)

    return inv_img


def image_to_kspace(im_tensor):
    # return (1,H,W,2)
    k_space_tensor = torch.rfft(im_tensor, 2, onesided=False)
    return k_space_tensor


# this is easy to control than gen_kspace_mask_deprecated
def gen_kspace_mask(shape, ratio, lowfreq_ratio):
    
    h, w = shape
    mask_low_freqs = int(np.ceil(int(lowfreq_ratio * h) / 2))
    frac = ratio - mask_low_freqs*2/h
    mask_fft = (np.random.RandomState(42).rand(h) < frac).astype(np.float32)
    mask_fft[:mask_low_freqs] = mask_fft[-mask_low_freqs:] = 1
    mask_fft = torch.from_numpy(mask_fft).view(1, h, 1, 1).expand(1,h,w,2)

    return mask_fft


def gen_kspace_mask_deprecated(shape, ratio, subsampling_type='lf_focused'):
    """ Create the mask that removes rows
        1 indicates the pixel is known during training.
        Expected shape is: 1 x height x width x 2
    """
    #pdb.set_trace()
    np.random.seed(0)
    img_mask_np = np.zeros(shape, np.float32) #img_fft_np.copy()
    nrows = shape[1]

    center_fraction = 0.1
    
    # Just simplfies the implementation
    if not float(ratio).is_integer():
        raise Exception("Subsampling ratio should be integral")

    # For random methods
    nrows_keep = int(nrows/ratio)
    inds = np.random.choice(nrows, size=nrows_keep, replace=False)

    # For center masking methods
    radius = int(nrows*center_fraction/2)
    middle = int(nrows/2)
    non_center_inds = (
        list(range(radius, middle-radius)) +
        list(range(middle+radius, nrows-radius)))
    non_center_subsample = list(set(inds) & set(non_center_inds))

    if subsampling_type == "random":
        img_mask_np[:,inds,:,:] = 1
    elif subsampling_type == "hf_focused":
        # region around center of image should be all 1
        # region around either end should be all 0
        img_mask_np[:,(middle-radius):(middle+radius),:,:] = 1
        img_mask_np[:,non_center_subsample,:,:] = 1
    elif subsampling_type == "lf_focused":
        # region around center of image should be all 0
        # region around either end should be all 1

        ## Want low frequences, which are near the top and bottom
        img_mask_np[:,:radius,:,:] = 1
        img_mask_np[:,(-radius):,:,:] = 1
        img_mask_np[:,non_center_subsample,:,:] = 1
    elif subsampling_type == "alternating_plus_hf":
        img_mask_np[:,::ratio,:,:] = 1
        img_mask_np[:,(middle-radius):(middle+radius),:,:] = 1
        img_mask_np[:,:radius,:,:] = 0
        img_mask_np[:,(-radius):,:,:] = 0
    elif subsampling_type == "alternating_plus_lf":
        #https://arxiv.org/pdf/1709.02576.pdf
        img_mask_np[:,::ratio,:,:] = 1
        img_mask_np[:,(middle-radius):(middle+radius),:,:] = 0
        img_mask_np[:,:radius,:,:] = 1
        img_mask_np[:,(-radius):,:,:] = 1
    elif subsampling_type == "alternating":
        # Really doesn'y work very well!
        img_mask_np[:,::ratio,:,:] = 1

    return img_mask_np


class FourierUtil():
    def __init__(self, unmask_ratio, normalize, low_freq_portion=0.8):
        # unmask_ratio: kspace overall mask ratio
        # normalize: if images to the input of _to_kspace is [-1, 1] normalized
        # low_freq_portion: what is the porportion of unmask_ratio to be low frequency

        self.unmask_ratio = unmask_ratio
        # self.mask_type = mask_type
        self.tanh_norm = normalize # if need to normalize to 0-1 (usually for gan)
        # we want low_freq_portion% are low frequency
        if low_freq_portion < 0.6 :
            print('Warning: low_frequency portion is low. It may causes images look bad')
        self.lowfreq_ratio = unmask_ratio * low_freq_portion
       
    def _to_kspace(self, img):

        if not hasattr(self, 'mask'):
            self.img_size = list(img.shape)
            ## old way in mri_deep_prior code
            # mask_np = gen_kspace_mask(self.img_size+[2], ratio=1//self.unmask_ratio, subsampling_type=self.mask_type)
            # self.mask = torch.from_numpy(mask_np)
            ## new way in anuroop's code
            self.mask = gen_kspace_mask(self.img_size[1:], ratio=self.unmask_ratio, lowfreq_ratio=self.lowfreq_ratio)
            

        if self.tanh_norm:
            # first put back image to [0-1]
            img = img.add(1).div(2)

        kspace_complex = image_to_kspace(img)
        # mask kspace
        masked_kspace = kspace_complex * self.mask

        rec_img = kspace_to_image(masked_kspace)
        #TODO do we need to recale rec_img
        rec_img.add_(-rec_img.min()).div_(rec_img.max() - rec_img.min())
        if self.tanh_norm:
            rec_img.mul_(2).add_(-1)
        
        _masked_kspace = kspace_complex * (1 - self.mask) # inverse of masked kspace
        _rec_img = kspace_to_image(_masked_kspace)
        _rec_img.add_(-_rec_img.min()).div_(_rec_img.max() - _rec_img.min())
        if self.tanh_norm:
            _rec_img.mul_(2).add_(-1)

        stacked_data = torch.cat([rec_img, _rec_img], dim=0) # 3x1xhxw

        return stacked_data
        

