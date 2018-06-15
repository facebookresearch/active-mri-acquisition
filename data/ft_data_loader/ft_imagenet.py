from PIL import Image
import os
import os.path
import numpy as np
import sys

import pickle
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from .ft_util import *
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    img_to_cat = {}
    with open(os.path.join(dir, 'labels.txt'), 'r') as f:
        for l in f.readlines():
            name, cat = l.split(',')
            img_to_cat[name] = cat.rstrip()
    
    cats = list(set(img_to_cat.values()))
    cat_to_class = {cats[i]: i for i in range(len(cats))}

    return img_to_cat, cat_to_class, cats


def make_dataset(dir, folder_to_cat, cat_to_class):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, cat_to_class[folder_to_cat[os.path.basename(root)]])
                    images.append(item)
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class FT_ImageNet(data.Dataset):
    
    def __init__(self, root, unmask_ratio, normalize, train=True, transform=None, target_transform=None,
                 loader=default_loader, download=False):
        folder_to_cat, cat_to_class, cats = find_classes(root)

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')

        imgs = make_dataset(root, folder_to_cat, cat_to_class)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))


        self.root = root
        self.imgs = imgs
        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        low_freq_portion = 0.8
        self.ft_util = FourierUtil(unmask_ratio, normalize=normalize, low_freq_portion=low_freq_portion)
        print('FT_ImageNet loader (keep ratio={:.2f}/{:.2f}): Found {} images in {} classes'.format(unmask_ratio, low_freq_portion, len(imgs), len(cats)))

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = img.convert('L')
        
        if self.transform is not None:
            img = self.transform(img)

         # transfer to fourier space
        kspace_data = self.ft_util._to_kspace(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, kspace_data

    def __len__(self):
        return len(self.imgs)

