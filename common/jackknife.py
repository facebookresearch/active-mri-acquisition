"""
Jackknife of reconstructions from compressed sensing, sampling rows in k-space.

This computes twice the sum of the leave-one-out errors of reconstructions from
compressed sensing, providing a jackknife estimate of the error.

Functions
---------
jackknife
    Plots twice the sum of the leave-one-out errors of reconstructions.
testjackknife
    Tests jackknife.
"""

import argparse
import logging
import math

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from . import ctorch
from . import admm


def jackknife(filein, fileout, mask, low, recon, recon_args, sdn=None):
    """
    plots twice the sum of the leave-one-out errors of reconstructions

    Plots to fileout twice the sum of the leave-one-out differences between
    the original image in filein and the reconstruction via recon applied to
    the k-space subsampling specified by mask (well, assuming mask includes
    all frequencies between -low and low), corrupting the k-space values with
    independent and identically distributed centered complex Gaussian noise
    whose standard deviation is sdn (sdn=0 if not provided explicitly).
    The "one" left out in the leave-one-out is a full row of k-space.

    The calling sequence of recon must be  (m, n, f, mask_th, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask_th = torch.from_numpy(mask.astype(np.unit8)).cuda(), and
    **recon_args is the unpacking of recon_args. The function recon must return
    a torch.Tensor (the reconstruction) and a float (the corresponding loss).

    N.B.: mask[-low+1], mask[-low+2], ..., mask[low-1] must be True.

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    mask : ndarray of bool
        indicators of whether to include (True) or exclude (False)
        the corresponding rows in k-space of the image from filein
    low : int
        bandwidth of low frequencies included in mask (between -low to low)
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0)

    Returns
    -------
    float
        loss for the reconstruction using all mask
    list of float
        losses for the reconstructions using all mask except for one row
    """
    # Set default parameters.
    if sdn is None:
        sdn = 0
    # Check that the mask includes the zero frequency.
    assert mask[0]
    # Check that the mask includes all low frequencies.
    for k in range(low):
        assert mask[k]
        assert mask[-k]
    # Read the image from disk.
    with Image.open(filein) as img:
        f_orig = np.array(img).astype(np.float64) / 255.
    m = f_orig.shape[0]
    n = f_orig.shape[1]
    # Fourier transform the image.
    ff_orig = np.fft.fft2(f_orig) / np.sqrt(m * n)
    # Add noise.
    ff_noisy = ff_orig.copy()
    ff_noisy += sdn * (np.random.randn(m, n) + 1j * np.random.randn(m, n))
    # Subsample the noisy Fourier transform of the original image.
    f = ctorch.from_numpy(ff_noisy[mask]).cuda()
    # Index the True values in mask (aside from the low frequencies).
    trues = []
    for k in range(mask.size):
        if k > low and k < m - low and mask[k]:
            trues.append(k)
    logging.info(
        'computing jackknife differences -- all {}'.format(len(trues)))
    # Perform the reconstruction using the entire mask.
    mask_th = torch.from_numpy(mask.astype(np.uint8)).cuda()
    reconf, lossf = recon(m, n, f, mask_th, **recon_args)
    reconf = reconf.cpu().numpy()
    # Perform the reconstruction omitting different samples in k-space.
    recons = np.ndarray((len(trues), m, n))
    loss = []
    for k in range(len(trues)):
        # Drop a row.
        mask1 = mask.copy()
        mask1[trues[k]] = False
        f1 = ctorch.from_numpy(ff_noisy[mask1]).cuda()
        # Reconstruct the image from the subsampled data.
        mask1_th = torch.from_numpy(mask1.astype(np.uint8)).cuda()
        recon1, loss1 = recon(m, n, f1, mask1_th, **recon_args)
        recon1 = recon1.cpu().numpy()
        # Record the results.
        recons[k, :, :] = recon1
        loss.append(loss1)
    # Calculate the sum of the leave-one-out differences.
    sumloo = np.sum(recons - reconf, axis=0)

    # Plot errors.
    # Set the numbers of rows and columns in the grid of plots.
    nr = 2
    nc = 2
    if m < n:
        plt.figure(figsize=(nc * 5, nr * 5 * m / n))
    else:
        plt.figure(figsize=(nc * 5 * n / m, nr * 5))
    # Remove the ticks and spines on the axes.
    matplotlib.rcParams['xtick.top'] = False
    matplotlib.rcParams['xtick.bottom'] = False
    matplotlib.rcParams['ytick.left'] = False
    matplotlib.rcParams['ytick.right'] = False
    matplotlib.rcParams['xtick.labeltop'] = False
    matplotlib.rcParams['xtick.labelbottom'] = False
    matplotlib.rcParams['ytick.labelleft'] = False
    matplotlib.rcParams['ytick.labelright'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.bottom'] = False
    matplotlib.rcParams['axes.spines.left'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    # Configure the colormaps.
    kwargs01 = dict(cmap='gray',
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    kwargs11 = dict(cmap='gray',
                    norm=matplotlib.colors.Normalize(vmin=-1, vmax=1))
    # Initialize numplot.
    numplot = 0
    # Plot the original.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('original')
    plt.imshow(f_orig, **kwargs01)
    # Plot the reconstruction using the full mask.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('recon')
    plt.imshow(reconf, **kwargs01)
    # Plot the difference from the original.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('error of recon')
    plt.imshow(reconf - f_orig, **kwargs11)
    # Plot twice the sum of the leave-one-out differences.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('twice sum of diffs')
    plt.imshow(sumloo * 2, **kwargs11)
    # Give the plots some space.
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    # Save the plots to disk.
    plt.savefig(fileout, bbox_inches='tight')

    return lossf, loss


def testjackknife(filein, fileout, subsampling_factor, recon, recon_args,
                  sdn=None):
    """
    tests jackknife

    Runs jackknife (and prints the losses) with filein, fileout, recon,
    recon_args, and sdn, creating a random mask that retains each row with
    probability subsampling_factor, supplemented by all rows between -sqrt(2m)
    and sqrt(2m), where filein contains an image with m rows.

    The calling sequence of recon must be  (m, n, f, mask_th, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask_th = torch.from_numpy(mask.astype(np.unit8)).cuda(), and
    **recon_args is the unpacking of recon_args. The function recon must return
    a torch.Tensor (the reconstruction) and a float (the corresponding loss).

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    subsampling_factor : float
        probability of retaining a row in the subsampling mask
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0 in jackknife)
    """
    # Obtain the size of the input image.
    with Image.open(filein) as img:
        n, m = img.size
    # Select which frequencies to retain.
    maski = set(
        np.floor(m * np.random.uniform(size=round(m * subsampling_factor))))
    mask = np.asarray([False] * m, dtype=bool)
    for i in maski:
        mask[int(i)] = True
    # Make the optimization well-posed by including the zero frequency.
    mask[0] = True
    # Include all low frequencies.
    low = round(math.sqrt(2. * m))
    for k in range(low):
        mask[k] = True
        mask[-k] = True
    # Generate jackknife plots.
    loss, losses = jackknife(filein, fileout, mask, low, recon, recon_args,
                             sdn)
    # Display the losses.
    print('loss = {}'.format(loss))
    print('losses = {}'.format(losses))


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    np.random.seed(seed=1337)
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filein',
        default='data/normal-mri-brain-with-mrv-teenager-1_128.png')
    parser.add_argument('--fileout', default='jackknife.png')
    parser.add_argument('--subsampling_factor', type=float, default=.5)
    parser.add_argument('--mu', type=float, default=1e12)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--sdn', type=float, default=0.02)
    args = parser.parse_args()
    # Run the test of jackknife.
    testjackknife(
        args.filein, args.fileout, args.subsampling_factor, admm.cs_fft,
        dict(mu=args.mu, beta=args.beta, n_iter=args.n_iter), args.sdn)
