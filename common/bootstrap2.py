"""
Bootstrap errors of reconstructions from compressed sensing, sampling radially.

This computes thrice the average of bootstrapped errors in reconstruction
from compressed sensing, providing a bootstrap estimate of the error.

N.B.: Whereas the sampling patterns in bootstrap.py appear often in practice,
the patterns from radialines.py used in this bootstrap2.py likely are relevant
only for prototyping.

Functions
---------
bootstrap2
    Plots thrice the average of bootstrapped errors in reconstruction.
testbootstrap2
    Tests bootstrap2.
"""

import argparse
import logging
import math

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

from . import ctorch
from . import admm2
from . import radialines


def bootstrap2(filein, fileout, subsampling_factor, angles, low, recon,
               recon_args, n_resamps=None, sdn=None):
    """
    plots thrice the average of bootstrapped errors in reconstruction

    Plots to fileout thrice the average of the differences between
    a reconstruction of the original image in filein and n_resamps bootstrap
    reconstructions via recon applied to the k-space subsamplings specified
    by angles fed into radialines and by other masks generated similarly
    (retaining each radial "line" with probability subsampling_factor, then
    adding all frequencies between -low to low in both directions), corrupting
    the k-space values with independent and identically distributed centered
    complex Gaussian noise whose standard deviation is sdn
    (sdn=0 if not provided explicitly).

    The calling sequence of recon must be  (m, n, f, mask, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask is the return from calls to radialines (with angles),
    supplemented by all frequencies between -low to low in both directions, and
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
        probability of retaining a radial "line" in the subsampling masks
    angles : list of float
        angles of the radial "lines" in the mask that radialines will construct
    low : int
        bandwidth of low frequencies included in mask (between -low to low
        in both the horizontal and vertical directions)
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    n_resamps : int, optional
        number of bootstrap resampled reconstructions (defaults to 100)
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0)

    Returns
    -------
    float
        loss for the reconstruction using the original angles
    list of float
        losses for the reconstructions using other, randomly generated masks
    """
    # Set default parameters.
    if n_resamps is None:
        n_resamps = 100
    if sdn is None:
        sdn = 0
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
    # Select which frequencies to retain.
    mask = radialines.randradialines(m, n, angles)
    # Include all low frequencies.
    for km in range(low):
        for kn in range(low):
            mask[km, kn] = True
            mask[m - 1 - km, kn] = True
            mask[km, n - 1 - kn] = True
            mask[m - 1 - km, n - 1 - kn] = True
    # Subsample the noisy Fourier transform of the original image.
    f = ctorch.from_numpy(ff_noisy[mask]).cuda()
    logging.info(
        'computing bootstrap2 resamplings -- all {}'.format(n_resamps))
    # Perform the reconstruction using the mask.
    reconf, lossf = recon(m, n, f, mask, **recon_args)
    reconf = reconf.cpu().numpy()
    # Fourier transform the reconstruction.
    freconf = np.fft.fft2(reconf) / np.sqrt(m * n)
    # Perform the reconstruction resampling new masks and samples in k-space.
    recons = np.ndarray((n_resamps, m, n))
    loss = []
    for k in range(n_resamps):
        # Select which frequencies to retain.
        angles1 = np.random.uniform(
            low=0, high=(2 * np.pi),
            size=round(2 * (m + n) * subsampling_factor))
        mask1 = radialines.randradialines(m, n, angles1)
        # Include all low frequencies.
        for km in range(low):
            for kn in range(low):
                mask1[km, kn] = True
                mask1[m - 1 - km, kn] = True
                mask1[km, n - 1 - kn] = True
                mask1[m - 1 - km, n - 1 - kn] = True
        # Subsample the Fourier transform of the reconstruction.
        f1 = ctorch.from_numpy(freconf[mask1]).cuda()
        # Reconstruct the image from the subsampled data.
        recon1, loss1 = recon(m, n, f1, mask1, **recon_args)
        recon1 = recon1.cpu().numpy()
        # Record the results.
        recons[k, :, :] = recon1
        loss.append(loss1)
    # Calculate the sum of the bootstrap differences.
    sumboo = np.sum(recons - reconf, axis=0)

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
    # Plot the reconstruction from the original angles provided.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('recon')
    plt.imshow(reconf, **kwargs01)
    # Plot the difference from the original.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('error of recon')
    plt.imshow(reconf - f_orig, **kwargs11)
    # Plot thrice the average of the bootstrap differences.
    numplot += 1
    plt.subplot(nr, nc, numplot)
    plt.title('thrice avg of diffs')
    plt.imshow(sumboo * 3 / n_resamps, **kwargs11)
    # Give the plots some space.
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    # Save the plots to disk.
    plt.savefig(fileout, bbox_inches='tight')

    return lossf, loss


def testbootstrap2(filein, fileout, subsampling_factor, recon, recon_args,
                   n_resamps=None, sdn=None):
    """
    tests bootstrap2

    Runs bootstrap2 (and prints the losses) with filein, fileout,
    subsampling_factor, recon, recon_args, n_resamps, and sdn, creating
    a random mask that retains each radial "line" with probability
    subsampling_factor.

    The calling sequence of recon must be  (m, n, f, mask, **recon_args),
    where filein contains an m x n image, f is the image in k-space subsampled
    to the mask, mask is the return from calls to radialines (with angles),
    and **recon_args is the unpacking of recon_args.
    The function recon must return a torch.Tensor (the reconstruction) and
    a float (the corresponding loss).

    Parameters
    ----------
    filein : str
        path to the file containing the image to be processed (the path may be
        relative or absolute)
    fileout : str
        path to the file to which the plots will be saved (the path may be
        relative or absolute)
    subsampling_factor : float
        probability of retaining a radial "line" in the subsampling masks
    recon : function
        returns the reconstructed image
    recon_args : dict
        keyword arguments for recon
    n_resamps : int, optional
        number of bootstrap resampled reconstructions (defaults to 100
        in bootstrap2)
    sdn : float, optional
        standard deviation of the noise to add (defaults to 0 in bootstrap2)
    """
    # Obtain the size of the input image.
    with Image.open(filein) as img:
        n, m = img.size
    # Select which frequencies to retain.
    angles = np.random.uniform(low=0, high=(2 * np.pi),
                               size=round(2 * (m + n) * subsampling_factor))
    # Generate bootstrap plots.
    low = 0
    loss, losses = bootstrap2(filein, fileout, subsampling_factor, angles, low,
                              recon, recon_args, n_resamps, sdn)
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
    parser.add_argument('--fileout', default='bootstrap2.png')
    parser.add_argument('--subsampling_factor', type=float, default=.1)
    parser.add_argument('--mu', type=float, default=1e12)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_resamps', type=int, default=100)
    parser.add_argument('--sdn', type=float, default=0.02)
    args = parser.parse_args()
    # Run the test of bootstrap2.
    testbootstrap2(
        args.filein, args.fileout, args.subsampling_factor, admm2.cs_fft,
        dict(mu=args.mu, beta=args.beta, n_iter=args.n_iter), args.n_resamps,
        args.sdn)
