# The DICOM dataset
# =================
#
# This file consists of two parts:
#
# 1. PyTorch datasets for accessing the DICOM dataset
# 2. A script that converts DICOM images into a binary file format
#
# ## 1. PyTorch Datasets

import os
import pathlib
import sys
import time
import json

import torch.utils.data as data
import torch
import numpy as np

# The `_DicomDataset` class is a thin wrapper over the memory mapped file. This
# class does not subclass PyTorch's `data.Dataset` for two reasons: (1) You
# shouldn't use this class directly when training (you should use `Slice` or
# `Volume` instead) and (2) I didn't want this class to be tied to PyTorch.

class _DicomDataset:
    """List-like access to the DICOM dataset.

    Args:
        root (string): Root directory path.

    Attributes:
        metadata (list): Information (for example, mean, std, and scan type) about the i-th volume.
        volumes (4d numpy array): The dimensions are <volume> x <slice> x <width> x <height>.
    """

    def __init__(self, root):
        self.metadata = json.load(open(os.path.join(root, 'metadata.json')))
        shape = len(self.metadata['volumes']), self.metadata['num_slices'], self.metadata['resolution'], self.metadata['resolution']
        self.volumes = np.memmap(os.path.join(root, 'data.bin'), self.metadata['dtype'], 'r').reshape(shape)

    def __getitem__(self, i):
        """
        Args:
            i (int): Index

        Returns:
            tuple (4d numpy array, dict): The first element contains the data, the second element contains metadata.
        """

        return self.volumes[i], self.metadata['volumes'][i]

    def __len__(self):
        return self.volumes.shape[0]

# Use the `Slice` class to train a model that works on slices (for example, all
# the ImageNet experiments would fall in this category).

class Slice(data.Dataset):
    """Access the DICOM dataset per slice.

    - Convert to float32.
    - Normalize the slice by the mean and standard deviation of the volume to which it belongs.
    - Compute the FFT of the slice and multiply the results with the mask.

    Args:
        mask_func(funciton): return the under-sampling mask. Will be called
        with arguments `(shape, args)`, expected to return a `torch.Tensor` of
        zeros and ones with shape `shape`.

        args(object):

        root (string): Root directory path.

        which (string): Which dataset to return. Choices are 'train', 'val',
        'public_leaderboard', and 'private_leaderboard'.

        resolution(string): Choices are 128 (for 128x128) and 320 (for 320x320).
    """

    def __init__(self, mask_func, args, which='train', resolution=None):
        if resolution is None:
            resolution = args.resolution
        self.mask_func = mask_func
        self.args = args
        self.dataset = _DicomDataset(pathlib.Path(args.dicom_root) / str(resolution) / which)
        self.num_slices = self.dataset.metadata['num_slices']

    def __getitem__(self, i):
        """
        Returns:
            tuple (input, target, mask, metadata): `input` has shape `1 x
            height x width x 2`, target has shape `1 x height x width`, mask
            has shape `1 x height x width x 2`, and metadata is a dictionary.

        """
        i = int(i)
        volume_i, slice_i = divmod(i, self.num_slices)
        volume, volume_metadata = self.dataset[volume_i]
        slice = volume[slice_i:slice_i + 1]
        slice = slice.astype(np.float32)
        slice = (slice - volume_metadata['mean']) / volume_metadata['std']
        slice = torch.from_numpy(slice)
        slice_fft = torch.rfft(slice, 2, normalized=True, onesided=False)
        mask = self.mask_func(slice_fft.shape, self.args)
        mask = torch.from_numpy(mask.astype(np.float32))
        slice_fft = slice_fft * mask
        return slice_fft, slice, mask, volume_metadata

    def __len__(self):
        return len(self.dataset) * self.num_slices


# Use the `Volume` class to work with volumes directly (for example, 3d
# convnets or 3d compressed sensing).

class Volume(data.Dataset):
    """Access the DICOM dataset per volume.

    The volume is normalized by the mean and standard deviation and converted
    to float32.

    Args:
        root (string): Root directory path.

    Attributes:
        dataset (_DicomDataset): The _DicomDataset that this class wraps.
        num_slices (int): Number of slices per volume.
    """

    def __init__(self, root):
        raise NotImplementedError
        self.dataset = _DicomDataset(root)

    def __getitem__(self, i):
        volume, volume_metadata = self.dataset[i]
        volume = volume.astype(np.float32)
        volume = (volume - volume_metadata['mean']) / volume_metadata['std']
        volume = torch.from_numpy(volume).unsqueeze(0)
        return volume

    def __len__(self):
        return len(self.dataset)

# 2. Convert DICOM files to mmaped file
# -------------------------------------
#
# Run this file as a script to convert all dicom files into one giant memory
# mapped file and one json file containing the metadata.

if __name__ == '__main__':
    import argparse
    import multiprocessing
    import random
    import re

    import pydicom
    import scipy.ndimage.interpolation
    import SimpleITK as sitk

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=128, help='Desired output spatial resolution')
    args = parser.parse_args()

# ### Reading dicom files
#
# I use SimpleITK to read a series of dicom files. SimpleITK is great because
# it knows how to order the slices in a series and compute pixel spacing and
# slice thickness.  However, I'm not that great at using SimpleITK to read
# attributes (I couldn't read the `ProcedureCodeSequence` attribute). I use
# pydicom, instead, to open the first dicom in a series and access its
# attributes.
#
# The slices all seem to be ordered in the same way, relative to the reference
# coordinate system. The following three figures from [Roni's
# blog](http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html)
# depict this.
#
# ![x](https://1.bp.blogspot.com/-3Z7kL7qKpJs/UbD9LNQNvSI/AAAAAAAAOSU/sXq82sHiCDM/s320/Patient-X.png)
# ![y](https://2.bp.blogspot.com/-VOL2o2GGEIE/UbD-_9e2e3I/AAAAAAAAOSk/WLPQ_Vm_Nsg/s320/Patient-Y.png)
# ![z](https://3.bp.blogspot.com/-lnlfTQiH1dc/UbD_yTPlO3I/AAAAAAAAOSw/T6RP6DF5RwE/s320/Patient-Z.png)

    def read_dicoms(pth):
        reader = sitk.ImageSeriesReader()
        for id in reader.GetGDCMSeriesIDs(str(pth)):
            fnames = reader.GetGDCMSeriesFileNames(str(pth), id)
            reader.SetFileNames(fnames)
            try:
                series = reader.Execute()
            except RuntimeError:
                continue
            dicom = pydicom.dcmread(str(fnames[0]))
            dicom.scan_type = get_scan_type(getattr(dicom, 'SeriesDescription', ''))
            dicom.pth = pth
            yield series, dicom

# ### Scan type
#
# Determine the type of scan from the series description. A typical series
# description would be something like `AX T2 FS`, which probably stands for
# axial T-2 weighted fat-suppressed image. We store only axial, sagittal, and
# coronal scans.
#
# ![sag_cor_ax](https://sites.google.com/a/wisc.edu/neuroradiology/_/rsrc/1468741099249/image-acquisition/the-basics/Even_Smaller.png)

    def get_scan_type(series_description):
        s = series_description.lower()
        if re.search(r'\b(l|r)?loc\b|local', s):
            return 'loc'
        elif 'shim' in s:
            return 'shim'
        elif 'scout' in s:
            return 'scout'
        elif 'scanogram' in s:
            return 'scanogram'
        elif 'petra' in s:
            return 'petra'
        elif 'survey' in s:
            return 'survey'
        elif 'sag' in s:
            return 'sag'
        elif 'cor' in s:
            return 'cor'
        elif 'ax' in s or 'tra' in s:
            return 'ax'
        return '?'

# We might also consider extracting other information from the series
# description, for example, whether the scan is T1-weighted, T2-weighted, or a
# proton density scan. The images look different depending on the type of scan.
# The image below shows a T1-weighted and a proton-density fat-suppressed
# image.
#
# ![t1_pd](https://med.stanford.edu/bmrgroup/Research/musculoskeletal-mri/_jcr_content/main/image.img.476.high.jpg)
#
# We would, currently, include both images.
#
# ### Filter
#
# We want to exclude some volumes from the dataset. For example, some volumes
# contain severe artifacts (see image below). But there are other reasons why a
# scan might be excluded.
#
# ![artefacts](http://www.peacemri.com/images/MAVRIC-NON_Knee.jpg)
#
# We accept a volume only if:
#
# - it's a sagittal, coronal, or axial scan and not, for example, a localization, scout, or shim scan.
# - it's not blacklisted
# - the patient is in the feet first-supine position
# - the number of slices is between 16 and 80
# - the slice thickness is between 1.5 mm and 6 mm
# - pixel spacing is the same in rows and columns and is between 0.1 mm and 0.9 mm
#
# The `blacklist` set contains series id's of scans with severe artifacts (this
# is not a complete list) so that we can skip them.

    blacklist = {
        '1.2.840.113654.2.70.1.332460465443223587862569843041671889028',
        '1.2.840.113654.2.70.1.196015218438253691092920732444512498677',
    }

    def print_skip_msg(dicom, msg):
        print(f'skip {dicom.SeriesInstanceUID} ({dicom.SeriesDescription}): {msg}')

    def filter(iterator):
        for series, dicom in iterator:
            spacing = series.GetSpacing()
            if dicom.scan_type not in {'sag', 'cor', 'ax'}:
                pass
            elif min(series.GetSize()[:2]) < args.resolution:
                print_skip_msg(dicom, f'image too small {series.GetSize()}')
            elif dicom.SeriesInstanceUID in blacklist:
                print_skip_msg(dicom, 'appears in blacklist')
            elif getattr(dicom, 'PatientPosition', '') != 'FFS':
                print_skip_msg(dicom, f'patient position {dicom.PatientPosition}')
            elif not 16 <= series.GetDepth() <= 80:
                print_skip_msg(dicom, f'number of slices {series.GetDepth()}')
            elif not 1.0 <= spacing[2] <= 6:
                print_skip_msg(dicom, f'slice thickness {spacing[2]}')
            elif spacing[0] != spacing[1]:
                print_skip_msg(dicom, f'pixel spacings differ {spacing[0]} != {spacing[1]}')
            elif not 0.1 <= spacing[0] <= 0.9:
                print_skip_msg(dicom, f'pixel spacing {spacing[0]}')
            else:
                yield series, dicom

# ### Resample
#
# The pixels spacing attribute of a dicom file specifies the physical size in
# mm of a pixel. We want to resample all volumes to the same pixel spacing. We
# also want to center crop the volumes to a common resolution. We define an
# affine transformation composed of the following three operations:
#
# 1. translate the volume so that its center is in the origin
# 2. scale the volume to the desired pixel spacing
# 3. translate the volume according to the desired output to achieve a center crop
#
# A single call to `scipy.ndimage.interpolation.affine_transform` is then used
# to scale and center-crop the volume.
#
# I know this is possible to accomplish with SimpleITK but I couldn't figure
# out how to do it.

    def affine3d_translate(d0, d1, d2):
        return np.array([[1, 0, 0, d0], [0, 1, 0, d1], [0, 0, 1, d2], [0, 0, 0, 1]])

    def affine3d_scale(s0, s1, s2):
        return np.array([[s0, 0, 0, 0], [0, s1, 0, 0], [0, 0, s2, 0], [0, 0, 0, 1]])

    def resample(volume, spacing, output_spacing, output_shape):
        '''Resize and center crop a 3d numpy array

        volume : 3d numpy array -- input volume
        spacing : 3-tuple -- pixel spacing of input volume
        output_spacing : 3-tuple -- desired pixel spacing of the output
        output_shape : 3-tuple -- desired size of the output
        '''
        q = np.array(output_spacing) / np.array(spacing)
        T = np.eye(4)
        T = T @ affine3d_translate(volume.shape[0] / 2, volume.shape[1] / 2, volume.shape[2] / 2)
        T = T @ affine3d_scale(q[0], q[1], q[2])
        T = T @ affine3d_translate(-output_shape[0] / 2, -output_shape[1] / 2, -output_shape[2] / 2)
        volume = scipy.ndimage.interpolation.affine_transform(volume, T, output_shape=output_shape)
        return volume


# ### Laterality
#
# Determine if it's a left knee or a right knee. The function returns either
# `'left'`, `'right'`, or `'unknown'`. When the function returns `'left'` or
# `'right'` it's not 100% accurate (for example, it fails on
# `dataset2/dicom/ST-4406922759858029449`), but it's not far from 100%. Another
# option would be to extract the laterality from the radiology report.
#
# protip: a good way to determine the laterality when looking at knee MRIs is to locate
# the fibula.
#
# ![fibula](https://www.knee-pain-explained.com/images/human-leg-knee-bones.jpg)

    def laterality(dicom):
        procedure = ''
        if hasattr(dicom, 'ProcedureCodeSequence'):
            procedure = dicom.ProcedureCodeSequence[0].CodeMeaning.split()[-1].lower()
        if procedure in {'left', 'right'}:
            return procedure
        elif getattr(dicom, 'Laterality', '') in {'L', 'R'}:
            return {'L': 'left', 'R': 'right'}[dicom.Laterality]
        return '?'

# ### Dataset Split
#
# We split the dataset into train (70%), val (10%), public_leaderboard (10%)
# and private_leaderboard (10%).  Whether a scan is placed in train, val,
# public_leaderboard, or private_leaderboard is determined by the patient id.
# We don't want to have, say, a T1 weighted image in the training set and a T2
# weighted image of the same patient in the validation set.

    def which_dataset(dicom):
        return random.Random(dicom.PatientID).choices(['train', 'val', 'public_leaderboard', 'private_leaderboard'], [70, 10, 10, 10])[0]

# ### Main
#
# Time to put everything together. The `process` function takes as input a
# path, for example
# `/scratch/jzb/knee_mri/dataset1/dicom/ST-791481395357892706`.  The dicoms in
# that path correspond to one patient's visit. During each visit a patient is scanned
# multiple times and a corresponding number of volumes is produced, typically
# between 5 and 10.
#
# The `process` function is also responsible for standardizing the values in each
# volume. That is, each output volume has zero mean and a standard deviation of
# one.

    def process(pth):
        for series, dicom in filter(read_dicoms(pth)):
            volume = sitk.GetArrayFromImage(series)
            if volume.dtype == np.int16 and volume.min() >= 0:
                volume = volume.astype(np.uint16)
            if volume.dtype != np.uint16:
                print_skip_msg(dicom, f'expected uint16, got {volume.dtype}')
                continue

            # resample
            output_pixel_spacing = [3, 128 / args.resolution, 128 / args.resolution]
            output_shape = [32, args.resolution, args.resolution]
            volume = resample(volume, series.GetSpacing()[::-1], output_pixel_spacing, output_shape)

            # write
            dir = pathlib.Path(f'tmp/mmap_tmp/{args.resolution}/{which_dataset(dicom)}/{pth.name}/{dicom.SeriesInstanceUID}/')
            dir.mkdir(parents=True)
            assert volume.shape == (32, args.resolution, args.resolution)
            assert volume.dtype == np.uint16
            volume.tofile(str(dir / 'volume.bin'))
            metadata = dict(
                patient_id=dicom.PatientID,
                accession_number=dicom.AccessionNumber,
                series_id=dicom.SeriesInstanceUID,
                scan_type=dicom.scan_type,
                pth=str(dicom.pth),
                series_description=dicom.SeriesDescription,
                pixel_spacing=output_pixel_spacing,
                laterality=laterality(dicom),
                mean=volume.mean(),
                std=volume.std(),
            )
            json.dump(metadata, open(dir / 'metadata.pkl', 'w'))

# Use Python's mutiprocessing module to parallelize the conversion. The script
# runs for about 15 minutes on our devfair machines (when resizing to a
# resolution of 128 and the input and output directories are on `/scratch`).

    paths = list(pathlib.Path('/scratch/jzb/knee_mri').glob('dataset*/dicom/ST-*'))
    pool = multiprocessing.Pool()
    pool.map(process, paths)

# The following code creates three giant memory-mappable files (one for train,
# one for val, and one for test)

    for root in pathlib.Path(f'tmp/mmap_tmp/{args.resolution}/').iterdir():
        dir = pathlib.Path(f'tmp/mmap/{args.resolution}/') / root.name
        dir.mkdir(exist_ok=True, parents=True)
        f = open(dir / 'data.bin', 'wb')
        volumes = []
        for visit in sorted(root.iterdir()):
            for volume in sorted(visit.iterdir()):
                volumes.append(json.load(open(volume / 'metadata.pkl', 'r')))
                f.write(open(volume / 'volume.bin', 'rb').read())
        metadata = dict(
            dtype='uint16',
            num_slices=32,
            resolution=args.resolution,
            volumes=volumes
        )
        json.dump(metadata, open(dir / 'metadata.json', 'w'))
