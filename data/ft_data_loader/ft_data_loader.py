"""
Source : https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py

Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import torch
import numpy as np
import os, sys
import PIL
# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from .ft_cifar10 import FT_CIFAR10
from .ft_imagenet import FT_ImageNet
from .ft_mnist import FT_MNIST
import random

def get_norm_transform(normalize):
    if normalize == 'gan':
        normalize_tf = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    elif normalize == 'zero_one':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
        )
    elif normalize == 'cae':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[1, 1, 1],
        )
    elif normalize == 'imagenet':  # channel = (channel - mean)/std
        normalize_tf = transforms.Normalize(
            mean=[0.43, 0.43, 0.43],
            std=[0.23, 0.23, 0.23],
        )
    return normalize_tf

# if fine_size is 128, load_size can be 144
def get_train_valid_loader(batch_size,
                           load_size, 
                           fine_size,
                           keep_ratio,
                           augment,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           normalize='gan',
                            which_dataset='MNIST',
                           data_dir='/private/home/sumanab/data/'    #'/private/home/zizhao/work/data/'
                         ):
    random_seed = 1234
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize_tf = get_norm_transform(normalize)
    # define transforms
    valid_transform = transforms.Compose(
        ([transforms.Grayscale()] if which_dataset == 'TinyImageNet' else []) + \
        [
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(fine_size),
            transforms.ToTensor(),
            normalize_tf,
        ])
    if augment:
        train_transform = transforms.Compose(
            ([transforms.Grayscale()] if which_dataset == 'TinyImageNet' else []) + \
            [
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.RandomCrop(fine_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_tf,
        ])
    else:
        train_transform = valid_transform

    print('load {} train/val (val ratio {:.4f}) dataset'.format(which_dataset, valid_size))
    if which_dataset in ('KNEE'):
         # a hacker way to import loader
        path = '/checkpoint/jzb/data/mmap'
        if not os.path.isdir(path):
            raise ImportError(path+' not exists. Download fast_mri_master repo and change the path')
        sys.path.insert(0, path)
        from common import args, dicom_dataset, subsample
        args = args.Args().parse_args(args=[])
        mask_func = subsample.Mask(reuse_mask=True)
        dataset = dicom_dataset.Slice(mask_func, args)
    elif which_dataset == 'KNEE_RAW':
        from .parallel_data_loader_raw import PCASingleCoilSlice, Mask
        mask_func = Mask(reuse_mask=False, subsampling_ratio=keep_ratio, random=True)
        root = '/private/home/zizhao/work/mri_data/multicoil/raw_mmap/FBAI_Knee/'
        if not os.path.isdir(root):
            raise ImportError(path+' not exists. Change to the right path.')
        dataset = PCASingleCoilSlice(mask_func, root, which='train')
        print(f'{which_dataset} train has {len(dataset)} samples')
        num_workers = 8
    elif which_dataset == 'KNEE_RAW_SINGLE_COIL':
        from .ft_util import RawSliceData, RawDataTransform, MaskFunc
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4]) #TODO: remove hardcoded value
        root = '/datasets01/fastMRI/112718'
        if not os.path.isdir(root):
            raise ImportError(root+' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func)
        train_dataset = RawSliceData(root + '/' + 'singlecoil_train', transform=data_transform, num_cols=368)  #TODO: remove hardcoded value
        valid_dataset = RawSliceData(root + '/' + 'singlecoil_val', transform=data_transform, num_cols=368)    #TODO: remove hardcoded value
        print(f'{which_dataset} train has {len(train_dataset)} train samples and {len(valid_dataset)} validation samples')
        num_workers = 8

        train_idx = list(range(len(train_dataset)))
        valid_idx = list(range(len(valid_dataset)))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SequentialSampler2(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )

        return train_loader, valid_loader

    elif which_dataset == 'TinyImageNet':
        train_dir = '/datasets01/tinyimagenet/081318/train'
        dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    else:
        if which_dataset == 'CIFAR10':
            dataset = FT_CIFAR10
            data_dir = '/private/home/sumanab/data/'    #'/private/home/zizhao/work/data/'
        elif which_dataset == 'ImageNet':
            dataset = FT_ImageNet
            data_dir = '/datasets01/imagenet_resized_144px/060718/061417'
        elif which_dataset == 'MNIST':
            dataset = FT_MNIST
        
        # load the dataset
        dataset = dataset(
            root=data_dir, train=True, normalize=normalize,
            download=True, transform=train_transform,  unmask_ratio=keep_ratio,
        )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler2(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    
    return (train_loader, valid_loader)

class SequentialSampler2(Sampler):
    r"""Samples elements sequentially, in the order of given list.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)

def get_test_loader(batch_size,
                    load_size,
                    fine_size,
                    keep_ratio,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=False,
                    normalize='gan',
                    which_dataset='MNIST',
                    data_dir='/private/home/sumanab/data/'  #'/private/home/zizhao/work/data/'
                    ):

    random_seed = 1234
    normalize_tf = get_norm_transform(normalize)
    # define transform
    transform = transforms.Compose(
        ([transforms.Grayscale()] if which_dataset == 'TinyImageNet' else []) + \
        [
            transforms.Resize(size=(load_size, load_size), interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(fine_size),
            transforms.ToTensor(),
            normalize_tf,
        ])

    print('load {} test dataset'.format(which_dataset))

    if which_dataset in ('KNEE'):
         # a hacker way to import loader
        sys.path.insert(0, '/checkpoint/jzb/data/mmap')
        from common import args, dicom_dataset, subsample
        args = args.Args().parse_args(args=[])
        mask_func = subsample.Mask(reuse_mask=True)
        args.subsampling_ratio = 1//keep_ratio
        print(f'KNEE >> subsampling_ratio: {args.subsampling_ratio}' )
        dataset = dicom_dataset.Slice(mask_func, args, which='val')
    elif which_dataset == 'KNEE_RAW':
        from .parallel_data_loader_raw import PCASingleCoilSlice, Mask
        sys.path.insert(0, '/private/home/zizhao/work/fast_mri_master')
        print(f'KNEE_RAW >> subsampling_ratio: {keep_ratio}' )
        mask_func = Mask(reuse_mask=True, subsampling_ratio=keep_ratio, random=False)
        root = '/private/home/zizhao/work/mri_data/multicoil/raw_mmap/FBAI_Knee/'
        dataset = PCASingleCoilSlice(mask_func, root, which='val')
        print(f'{which_dataset} val has {len(dataset)} samples')
        num_workers = 8
    elif which_dataset == 'KNEE_RAW_SINGLE_COIL':
        from .ft_util import RawSliceData, RawDataTransform, MaskFunc
        mask_func = MaskFunc(center_fractions=[0.125], accelerations=[4]) #TODO: remove hardcoded value
        root = '/datasets01/fastMRI/112718'
        if not os.path.isdir(root):
            raise ImportError(root+' not exists. Change to the right path.')
        data_transform = RawDataTransform(mask_func)
        dataset = RawSliceData(root + '/' + 'singlecoil_test', transform=data_transform, num_cols=368)  #TODO: remove hardcoded value
    elif which_dataset == 'TinyImageNet':
        test_dir = '/datasets01/tinyimagenet/081318/test'
        dataset = datasets.ImageFolder(test_dir, transform=transform)
    else:  
        if which_dataset == 'CIFAR10':
            dataset = FT_CIFAR10
        elif which_dataset == 'ImageNet':
            dataset = FT_ImageNet
            data_dir = '/datasets01/imagenet_resized_144px/060718/061417'
        elif which_dataset == 'MNIST':
            dataset = FT_MNIST
    
        dataset = dataset(
            root=data_dir, train=False, normalize=normalize,
            transform=transform,  unmask_ratio=keep_ratio,
        )

    if shuffle:
        # TODO these seed setting may not really useful
        # torch.manual_seed(random_seed)
        # torch.cuda.manual_seed_all(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        # torch.backends.cudnn.deterministic = True

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        test_sampler = SequentialSampler2(indices)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, sampler=test_sampler, drop_last=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )

    return data_loader

