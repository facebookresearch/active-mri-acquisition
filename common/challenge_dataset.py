# The Challenge Dataset
# =====================
#
# A script that generates the competition dataset and a class that reads slices from the dataset.

import pathlib
import sys
import time
import traceback

import pylab as plt
from numpy.fft import fftshift, ifftshift, ifft2, fft2
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

from common import args as args_module, transforms, ctorch


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, ground_truth, inner_slice_count, sample_rate=1, resolution_filter=None, fnames=None):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if ground_truth not in ('esc', 'rss'):
            raise ValueError('ground_truth should be either "esc" or "rss"')

        self.transform = transform
        self.recons_key = "reconstruction_" + ground_truth
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            if fnames is not None and str(fname) not in fnames:
                continue

            if resolution_filter is None or kspace.shape[1:] == resolution_filter:
                self.examples += [(fname, slice) for slice in range(num_slices)
                        if slice > num_slices // 2 - inner_slice_count // 2 and slice < num_slices // 2 + inner_slice_count // 2]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)


# Generate Competition Dataset
# ----------------------------
#
# Calling this file as a script will generate the competition dataset. This
# involves the following steps:
#
# - Read ismrmr file.
# - Compute ground truth reconstruction.
# - Compute emulated single coil kspace.
# - Split the dataset into train, val, test, and challenge.
# - Store each volume as a single .h5 file.

if __name__ == '__main__':
    import argparse
    import hashlib
    import os
    import pickle
    import random
    import shutil
    import collections
    import datetime
    import hashlib

    import ismrmrd
    import skimage.filters
    import scipy.optimize

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=pathlib.Path, default='/checkpoint/jzb/old_checkpoint02/data/nyu/raw_data')
    parser.add_argument('--output-dir', type=pathlib.Path, default='/checkpoint/jzb/data/challenge_tmp')
    args = parser.parse_args()

    # emulated single-coil
    def esc(kspace, coil_imgs, recon_rss):
        def func(x):
            c = x.size // 2
            x = ctorch.ComplexTensor(torch.from_numpy(x[:c]), torch.from_numpy(x[c:])).requires_grad_()
            loss = torch.sum((torch.sqrt(abs(A @ x)) - torch.sqrt(b.real))**2)
            loss.backward()
            grad = ctorch.ComplexTensor(x.real.grad, x.imag.grad).numpy()
            grad = np.concatenate((grad.real, grad.imag))
            return loss.item(), grad

        mask = np.zeros(recon_rss.shape, dtype=bool)
        d1 = (mask.shape[1] - 320) // 2
        d2 = (mask.shape[2] - 320) // 2
        mask[:, d1:d1+320, d2:d2+320] = True
        A = coil_imgs.transpose(0, 2, 3, 1)[mask]
        b = recon_rss[mask]

        scale = 1e3 / np.sqrt(np.linalg.norm(b))
        A *= scale
        b *= scale

        x = np.linalg.lstsq(A, b, rcond=None)[0]
        x = np.concatenate((x.real, x.imag))
        A, b = ctorch.from_numpy(A), ctorch.from_numpy(b)
        x, loss, _ = scipy.optimize.fmin_l_bfgs_b(func, x, iprint=-1)
        x = x[:x.size//2] + 1j * x[x.size//2:]
        kspace_esc = np.sum(kspace * x[None, :, None, None], axis=1)
        recon_esc = np.sum(coil_imgs * x[None, :, None, None], axis=1)
        return kspace_esc, recon_esc

    def process(filename):
        t_process = time.time()
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        enc = hdr.encoding[0]
        enc_size = (enc.encodedSpace.matrixSize.x,
                    enc.encodedSpace.matrixSize.y,
                    enc.encodedSpace.matrixSize.z)
        rec_size = (enc.reconSpace.matrixSize.x,
                    enc.reconSpace.matrixSize.y,
                    enc.reconSpace.matrixSize.z)
        enc_limits_min = enc.encodingLimits.kspace_encoding_step_1.minimum
        enc_limits_max = enc.encodingLimits.kspace_encoding_step_1.maximum + 1
        enc_limits_center = enc.encodingLimits.kspace_encoding_step_1.center
        num_acqs = dset.number_of_acquisitions()
        num_slices = enc.encodingLimits.slice.maximum + 1
        num_coils = hdr.acquisitionSystemInformation.receiverChannels
        num_reps = enc.encodingLimits.repetition.maximum + 1
        num_contrasts = enc.encodingLimits.contrast.maximum + 1
        seed_mask = tuple(map(ord, filename.stem + '_mask'))
        seed_dataset = tuple(map(ord, patient_id[filename.stem] + '_dataset'))
        seed_acceleration = tuple(map(ord, filename.stem + '_acceleration'))

        assert 200 <= enc_limits_max < 1000
        assert num_coils == 15
        assert rec_size == (320, 320, 1)
        assert enc_limits_min == 0
        assert num_reps == 1
        assert num_contrasts == 1

        # read kspace
        kspace = np.zeros((num_slices, num_coils, enc_size[0], enc_limits_max), dtype=np.complex128)
        xs = [[] for _ in range(num_slices)]
        position = [None] * num_slices
        slice_dir = [None]
        for i in range(num_acqs):
            acq = dset.read_acquisition(i)
            if acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                continue
            if acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                continue
            acq.clear_flag(ismrmrd.ACQ_FIRST_IN_SLICE)
            acq.clear_flag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            acq.clear_flag(ismrmrd.ACQ_LAST_IN_REPETITION)
            acq.clear_flag(ismrmrd.ACQ_LAST_IN_SLICE)
            assert acq.flags == 0
            assert acq.data.dtype == np.complex64
            assert acq.idx.kspace_encode_step_2 == 0
            xs[acq.idx.slice].append(acq.idx.kspace_encode_step_1)
            def assign_or_assert_equal(lst, idx, val):
                if lst[idx] is None:
                    lst[idx] = val
                else:
                    np.testing.assert_array_equal(lst[idx], val)
            assign_or_assert_equal(slice_dir, 0, np.array(acq.slice_dir))
            assign_or_assert_equal(position, acq.idx.slice, np.array(acq.position))
            kspace[acq.idx.slice, :, :, acq.idx.kspace_encode_step_1] = acq.data

        for slice in range(num_slices):
            assert sorted(set(xs[slice])) == list(range(enc_limits_max))

        # reorder the slices
        proj = [np.dot(slice_dir[0], p) for p in position]
        slice_order = np.argsort(proj)
        kspace = kspace[slice_order]

        # zero-pad kspace
        padding_left = enc_size[1] // 2 - enc_limits_center
        kspace_padded = np.zeros((num_slices, num_coils, enc_size[0], enc_size[1]), dtype=np.complex128)
        kspace_padded[:, :, :, padding_left:padding_left + kspace.shape[3]] = kspace
        kspace = kspace_padded

        coil_imgs = transforms.cifft2(kspace)

        # root-sum-of-squares reconstruction
        recon_rss = np.sqrt(np.sum(np.abs(coil_imgs)**2, axis=1))

        # emulated single-coil
        kspace_esc, recon_esc = esc(kspace, coil_imgs, recon_rss)
        # kspace_esc, recon_esc = kspace, recon_rss

        # generate mask
        def get_mask(num_cols, acceleration, center_fraction, seed):
            num_lf = int(round(num_cols * center_fraction))
            p = (num_cols / acceleration - num_lf) / (num_cols - num_lf)
            mask = np.random.RandomState(seed).uniform(size=num_cols) < p
            pad = (num_cols - num_lf + 1) // 2
            mask[pad:pad + num_lf] = True
            return mask, num_lf
        if random.Random(seed_acceleration).random() < 0.5:
            mask_kwargs = dict(acceleration=4, center_fraction=0.08)
        else:
            mask_kwargs = dict(acceleration=8, center_fraction=0.04)
        mask, num_lf = get_mask(kspace.shape[3], seed=seed_mask, **mask_kwargs)
        kspace_masked = kspace * mask[None, None, None,:]
        kspace_esc_masked = kspace_esc * mask[None, None, :]

        # crop reconstructions
        recon_rss_cropped = np.abs(transforms.center_crop(recon_rss, (320, 320)))
        recon_esc_cropped = np.abs(transforms.center_crop(recon_esc, (320, 320)))

        # save files
        def write_metadata_train(f, which):
            assert which in {'singlecoil', 'multicoil'}
            recon = recon_rss_cropped if which == 'multicoil' else recon_esc_cropped
            f.attrs['norm'] = np.linalg.norm(recon)
            f.attrs['max'] = np.max(recon)
            write_metadata(f)

        def write_metadata_test(f):
            f.attrs['acceleration'] = mask_kwargs['acceleration']
            f.attrs['num_low_frequency'] = num_lf
            write_metadata(f)

        def write_metadata(f):
            f.attrs['patient_id'] = hashlib.sha3_256(patient_id[filename.stem].encode()).hexdigest()
            f.attrs['acquisition'] = acquisition[filename.stem]
            f.create_dataset('ismrmrd_header', data=dset.read_xml_header())

        datasets = [
            ('train', 0.625),
            ('val', 0.125),
            ('multicoil_test', 0.0625),
            ('singlecoil_test', 0.0625),
            ('multicoil_challenge', 0.0625),
            ('singlecoil_challenge', 0.0625),
        ]
        d_names, d_probs = zip(*datasets)
        which_dataset = random.Random(seed_dataset).choices(d_names, d_probs)[0]

        if which_dataset in {'train', 'val'}:
            f = h5py.File(args.output_dir / f'public/multicoil_{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_train(f, 'multicoil')
            f.create_dataset('kspace', data=kspace.astype(np.complex64))
            f.create_dataset('reconstruction_rss', data=recon_rss_cropped.astype(np.float32))
            f = h5py.File(args.output_dir / f'public/singlecoil_{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_train(f, 'singlecoil')
            f.create_dataset('kspace', data=kspace_esc.astype(np.complex64))
            f.create_dataset('reconstruction_esc', data=recon_esc_cropped.astype(np.float32))
            f.create_dataset('reconstruction_rss', data=recon_rss_cropped.astype(np.float32))
        elif which_dataset in {'multicoil_test', 'multicoil_challenge'}:
            f = h5py.File(args.output_dir / f'public/{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_test(f)
            f.create_dataset('kspace', data=kspace_masked.astype(np.complex64))
            f.create_dataset('mask', data=mask)
            f = h5py.File(args.output_dir / f'private/{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_test(f)
            f.create_dataset('reconstruction_rss', data=recon_rss_cropped.astype(np.float32))
            f.create_dataset('mask', data=mask)
        elif which_dataset in {'singlecoil_test', 'singlecoil_challenge'}:
            f = h5py.File(args.output_dir / f'public/{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_test(f)
            f.create_dataset('kspace', data=kspace_esc_masked.astype(np.complex64))
            f.create_dataset('mask', data=mask)
            f = h5py.File(args.output_dir / f'private/{which_dataset}/{filename.stem}.h5', 'w')
            write_metadata_test(f)
            f.create_dataset('reconstruction_esc', data=recon_esc_cropped.astype(np.float32))
            f.create_dataset('reconstruction_rss', data=recon_rss_cropped.astype(np.float32))
            f.create_dataset('mask', data=mask)
        else:
            assert False
        print(filename.stem, int(time.time() - t_process))

    os.makedirs(args.output_dir / 'public/multicoil_train', exist_ok=True)
    os.makedirs(args.output_dir / 'public/multicoil_val', exist_ok=True)
    os.makedirs(args.output_dir / 'public/multicoil_test', exist_ok=True)
    os.makedirs(args.output_dir / 'public/multicoil_challenge', exist_ok=True)
    os.makedirs(args.output_dir / 'public/singlecoil_train', exist_ok=True)
    os.makedirs(args.output_dir / 'public/singlecoil_val', exist_ok=True)
    os.makedirs(args.output_dir / 'public/singlecoil_test', exist_ok=True)
    os.makedirs(args.output_dir / 'public/singlecoil_challenge', exist_ok=True)
    os.makedirs(args.output_dir / 'private/multicoil_test', exist_ok=True)
    os.makedirs(args.output_dir / 'private/multicoil_challenge', exist_ok=True)
    os.makedirs(args.output_dir / 'private/singlecoil_test', exist_ok=True)
    os.makedirs(args.output_dir / 'private/singlecoil_challenge', exist_ok=True)

    # read patient_ids
    patient_id = {}
    acquisition = {}
    for line in (args.data_dir / 'map_files/dataset_key.txt').open():
        fname, id, acq = line.strip().split('#')
        patient_id[fname] = id
        acquisition[fname] = acq

    slurm_ntasks = int(os.getenv('SLURM_NTASKS', '1'))
    slurm_procid = int(os.getenv('SLURM_PROCID', '0'))
    filenames = sorted(list(args.data_dir.glob('*/FBAI_Knee/*')))
    filenames = filenames[slurm_procid:len(filenames):slurm_ntasks]
    for filename in filenames:
        try:
            process(filename)
        except:
            print('-' * 60)
            print(filename)
            traceback.print_exc(file=sys.stdout)
