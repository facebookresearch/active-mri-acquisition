import torch
import torch.nn as nn
import torchvision
import sys

from torch.autograd import Variable
import numpy as np
from PIL import Image
import PIL
import numpy as np
import pdb

import matplotlib.pyplot as plt

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_optimizer(p, args):
    if args.optimizer == "adam":
        # Default is 0.9, 0.999, so we kind of immitate that
        if "momentum2" in args.__dict__:
            beta2 = args.momentum2
        else:
            beta2 = 1-((1-args.momentum)/100)
        optimizer = torch.optim.Adam(p, lr=args.initial_lr, betas=(args.momentum, beta2),
            weight_decay=args.decay, eps=args.adam_eps)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(p, lr=args.initial_lr, momentum=args.momentum,
            weight_decay=args.decay)
    elif args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(p, lr=args.initial_lr, momentum=args.momentum,
            weight_decay=args.decay)
    else:
        raise Exception("Unrecognised optimizer name")
    return optimizer

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Variable that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation=None):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np)+factor,12+factor))
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1,2,0), interpolation=interpolation)
    plt.show()

    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Variable of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = Variable(torch.zeros(shape))

        fill_noise(net_input.data, noise_type)
        net_input.data *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_var(meshgrid)
    else:
        assert False

    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32)/255.

def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1,2,0)

    return Image.fromarray(ar)

def save_np_img(img_np, path, img_mean=0, img_sd=1, normalize=False):
    ''' Saves a np.array as a monochrome image. Img is unnormalized with the passed mean and sd.
        Scaling should be [0,1] input unless normalize=T, in which case it's normalized by empirical min/max.
    '''
    # Rescale to 0-255
    if normalize:
        img_min = np.min(np.real(img_np))
        img_max = np.max(np.real(img_np))
        ar = (img_np-img_min)/(img_max-img_min)
    else:
        ar = img_np*img_sd + img_mean

    ar = np.abs(ar)

    # Rescale to 0-255
    #pdb.set_trace()
    ar = np.clip(ar*255,0,255).astype(np.uint8)

    # Drop initial len 1 dimensions
    if ar.shape[0] == 1:
        ar = ar[0]
    if ar.shape[0] == 1:
        ar = ar[0]
    Image.fromarray(ar).save(path)

def kspace_flip_quarters(kimg):
    ''' move high frequences to the outside of the image instead of the middle
        this format is more common than the numpy/pytorch default
    '''
    #pdb.set_trace()
    if  kimg.shape[0] % 2 != 0 or  kimg.shape[1] % 2 != 0:
        raise Exception("Only even dimension images are supported")

    rowmid = int(kimg.shape[0]/2)
    colmid = int(kimg.shape[1]/2)

    timg = kimg.copy()
    #pdb.set_trace()
    timg[:rowmid,:colmid] = np.flip(np.flip(timg[:rowmid,:colmid], 0), 1)
    timg[rowmid:,:colmid] = np.flip(np.flip(timg[rowmid:,:colmid], 0), 1)
    timg[:rowmid,colmid:] = np.flip(np.flip(timg[:rowmid,colmid:], 0), 1)
    timg[rowmid:,colmid:] = np.flip(np.flip(timg[rowmid:,colmid:], 0), 1)

    return timg

def save_np_kspace_img(k_np, path, brighten=0.8, scaling=1.0, flip=True):
    ''' Does log scaling specific to kspace input before saving as a image
    '''

    if k_np.shape[-1] == 2:
        k_np_abs = np.sqrt(k_np[...,0]**2 + k_np[...,1]**2)
    else:
        k_np_abs = np.abs(k_np)

    # Drop initial dimension if it's 1
    if k_np_abs.shape[0] == 1:
        k_np_abs = k_np_abs[0]


    if flip:
        k_np_abs = np.fft.fftshift(k_np_abs)
        #k_np_abs = kspace_flip_quarters(k_np_abs)

    ar = np.log(1 + scaling*k_np_abs)
    ar = ar/np.max(ar)
    ar = ar**(brighten) # matlab code uses 0.2? Seems high

    save_np_img(ar, path)

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype = torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

def tensor_to_np(img_tensor):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_tensor.cpu().numpy()[0]

def asscalar(tensor):
    '''Converts a 1D tensor to a python scalar value
    '''
    return tensor.item()

def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Variables to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)

        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


#https://github.com/mrirecon/bart/blob/master/python/cfl.py
# For the BART tool
def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F') # column-major


def writecfl(name, array):
    fname = name + ".hdr"
    h = open(fname, "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()
    print(f"Wrote {fname} to disk")

def save_fft(fname, x):
    img = Image.fromarray((x*255).astype(np.uint8))
    img.save(fname)
    print(f"Saved {fname}")
