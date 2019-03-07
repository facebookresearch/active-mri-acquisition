import os
import warnings
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import logging
from torch.autograd import Variable
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import pdb

def imagenet(args, load_train=True, load_test=True):
    """
        requires args.batch_size to be set. Sets args.img_mean, args.img_std, args.nbatches, args.ndata
    """
    if 'num_workers' not in args.__dict__:
        args.num_workers = 8
    args.img_mean = 0.43
    args.img_std = 0.23

    # You can't use imagenet without these really
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # These versions have been pre-resized to ~144, for performance
    train_dir = "/datasets01/imagenet_resized_144px/060718/061417/train"
    test_dir = "/datasets01/imagenet_resized_144px/060718/061417/val"
    #train_dir = "/datasets01/imagenet_full_size/061417/train"
    #test_dir = "/datasets01/imagenet_full_size/061417/val"

    # Missing EXIF data creates annoying warnings from imagenet, which this suppresses
    warnings.filterwarnings('ignore', module=".*TiffImagePlugin.*")

    transform_train = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(144),
        #transforms.CenterCrop(128), #Debugging
        transforms.RandomCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[args.img_mean], std=[args.img_std])
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[args.img_mean], std=[args.img_std])
    ])

    if load_train:
        logging.info(f"Walking training data directory {train_dir} ...")
        train_dataset = datasets.ImageFolder(train_dir, transform_train)

        logging.info(f"Loader ... ({train_dir})")

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=getattr(args, "shuffle", True),
            num_workers=args.num_workers, pin_memory=True, drop_last=True)

        args.ndata = len(train_loader.dataset)
        args.nbatches = len(train_loader) # Fairly slow

        logging.info(f"Train Loader created, nbatches: {args.nbatches} ndata: {args.ndata}")
    else:
        train_loader = None

    if load_test:
        logging.info(f"Walking test data directory {test_dir} ...")
        test_dataset = datasets.ImageFolder(test_dir, transform_test)

        logging.info(f"Loader ... ({test_dir})")

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        args.test_ndata = len(test_loader.dataset)
        args.test_nbatches = len(test_loader) # Fairly slow

        logging.info(f"Test Loader created, nbatches: {args.test_nbatches} ndata: {args.test_ndata}")
    else:
        test_loader = None

    return train_loader, test_loader
