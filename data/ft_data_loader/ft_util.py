import numpy as np
# from numpy.fft import fftshift, ifftshift, fftn, ifftn
import torch
import h5py
import pathlib
from torch.utils.data import Dataset, DataLoader


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


class MaskFunc:
    def __init__(self, center_fractions, accelerations):
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        num_high_freqs = num_cols // acceleration - num_low_freqs
        mask = self.create_lf_focused_mask(num_cols, num_high_freqs, num_low_freqs)
        mask = np.fft.ifftshift(mask, axes=0)

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-1] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask

    def create_lf_focused_mask(self, num_cols, num_high_freqs, num_low_freqs):
        p = num_high_freqs / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < p
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True
        return mask


def roll(x, shift, dim):
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def ifftshift(x, dim=None):
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


class RawDataTransform:
    def __init__(self, mask_func, seed=None):
        self.mask_func = mask_func
        self.seed = seed

    def __call__(self, kspace, attrs):
        kspace = torch.from_numpy(np.stack([kspace.real, kspace.imag], axis=-1))
        kspace = ifftshift(kspace, dim=(0, 1))
        image = torch.ifft(kspace, 2, normalized=False)
        image = ifftshift(image, dim=(0, 1))
        norm = torch.sqrt(image[..., 0] ** 2 + image[..., 1] ** 2).max()
        image /= norm
        kspace /= norm
        shape = np.array(kspace.shape)
        mask = self.mask_func(shape, self.seed)
        return mask, image, kspace

    def postprocess(self, data, hps):
        mask, inputs, kspace = data
        mask = data[0].repeat(1, 1, kspace.shape[1], 1)
        inputs = inputs.cuda().permute(0, 3, 1, 2)
        kspace = kspace.cuda().unsqueeze(1)
        mask = mask.cuda()
        mask_k_space = mask
        mask = mask.unsqueeze(4).repeat(1, 1, 1, 1, 2)
        masked_kspace = kspace * mask
        masked_image = ifftshift(torch.ifft(masked_kspace, 2, normalized=False), dim=(2, 3))
        masked_image = masked_image.squeeze(1).permute(0, 3, 1, 2)
        return inputs, masked_image, mask_k_space


# Load k-space data
class RawSliceData(Dataset):
    def __init__(self, root, transform, num_cols=None):
        self.transform = transform
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        for fname in sorted(files):
            data = h5py.File(fname, 'r')
            if num_cols is not None and data['kspace'].shape[2] != num_cols:
                continue
            kspace = data['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            return self.transform(kspace, data.attrs)
        

