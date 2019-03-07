"""Compressed sensing for images sampled at an arbitrary subset of k-space.

This implements minimal total-variation reconstruction of a two-dimensional
image via Algorithm 1 at the end of Section 2.2 of "Alternating direction
algorithms for total variation deconvolution in image reconstruction"
by Min Tao and Junfeng Yang, which is available from Optimization Online at
http://www.optimization-online.org/DB_HTML/2009/11/2463.html
The functions below refer to this article as "Tao-Yang."

N.B.: Whereas the sampling patterns in admm.py appear often in actual practice,
the more general patterns of this admm2.py are relevant only for prototyping.

Functions
---------
zero_padded
    Supplements the Fourier-domain input with zeros to fill out a full image.
adm
    ADMM iterations for an l_2 group lasso.
cs_baseline
    Recovers an image from a subset of its frequencies without any FFTs.
cs_fft
    Recovers an image from a subset of its frequencies using FFTs.
runtestadmm
    Tests cs_baseline or cs_fft as specified.
"""

import argparse
import math
import time

import matplotlib
matplotlib.use('agg')
import pylab as plt
from PIL import Image
import numpy as np
import scipy.linalg
import torch

from . import ctorch


def zero_padded(m, n, x, mask):
    """
    Supplements the Fourier-domain input with zeros to fill out a full image.

    Pads x with zeros to form a full m x n image, with the possibly nonzero
    locations specified by mask. Then applies the (two-dimensional) unitary
    inverse discrete Fourier transform.

    Parameters
    ----------
    m : int
        number of rows in the image being formed
    n : int
        number of columns in the image being formed
    x : numpy.ndarray or ctorch.ComplexTensor
        potentially nonzero entries (prior to the inverse Fourier transform)
    mask : numpy.ndarray or torch.LongTensor
        indicators of the positions of the entries of x in the full m x n array

    Returns
    -------
    numpy.ndarray or ctorch.ComplexTensor
        inverse Fourier transform of x padded with zeros
    """
    if isinstance(x, np.ndarray):
        # Pad x with zeros, obtaining y.
        y = np.zeros((m, n), dtype=np.complex128)
        y[mask] = x
        # Scale the FFTs to make them unitary.
        return np.fft.ifft2(y) * np.sqrt(m * n)
    elif isinstance(x, ctorch.ComplexTensor):
        # Pad x with zeros, obtaining y.
        y = ctorch.ComplexTensor(x.real.new(m, n), x.imag.new(m, n)).zero_()
        mask_flat = mask[:, 0] * n + mask[:, 1]
        y.view(-1)[mask_flat] = x
        # Scale the FFTs to make them unitary.
        return ctorch.ifft2(y) * np.sqrt(m * n)
    else:
        raise TypeError('Input must be a numpy.ndarray ' +
                        'or a ctorch.ComplexTensor.')


def adm(f, K, D, mu, beta, n_iter):
    """
    ADMM iterations for an l_2 group lasso.

    Runs n_iter iterations of the alternating direction method of multipliers
    (ADMM) with coupling beta to minimize over x the following objective:
    mu |Kx - f|_2^2 / 2 + sum_i sqrt((Dx)_{2i}^2 + (Dx)_{2i+1}^2)
    where |Kx - f|_2^2 denotes the square of the Euclidean norm of Kx-f.

    Parameters
    ----------
    f : ndarray
        dependent variables in the least-squares regression
    K : ndarray
        independent variables (design matrix) in the least-squares regression
    D : ndarray
        matrix in the regularization term sqrt((Dx)_{2i}^2 + (Dx)_{2i+1}^2)
    mu : float
        regularization parameter
    beta : float
        coupling parameter for the ADMM iterations
    n_iter : int
        number of iterations to conduct

    Returns
    -------
    ndarray
        argmin[ mu |Kx - f|_2^2 / 2 + sum_i sqrt((Dx)_{2i}^2 + (Dx)_{2i+1}^2) ]
        where |Kx - f|_2^2 denotes the square of the Euclidean norm of Kx-f
    float
        objective value at the end of the ADMM iterations
    """
    assert type(f) == type(K)
    assert type(f) == type(D)
    n = K.shape[1]
    # Compute the upper triangular factor U in the Cholesky decomposition of
    # D^* D + mu / beta * K^* K.
    U = scipy.linalg.lu_factor(D.conj().T @ D + (mu / beta) * (K.conj().T @ K))
    # Apply (mu / beta) * K^* to the input f, obtaining Ktf.
    Ktf = (mu / beta) * (K.conj().T @ f)
    # Initialize the primal (x) and dual (la) solution vectors to zeros.
    x = np.zeros(n)
    la = np.zeros(2 * n)
    # Conduct iterations of alternating minimization.
    for i in range(n_iter):
        # Apply shrinkage via formula (2.7) from Tao-Yang, dividing both
        # arguments of the "max" operator in formula (2.7) by the denominator
        # of the rightmost factor in formula (2.7).
        a = (D @ x + la / beta).reshape(2, n)
        b = scipy.linalg.norm(a, axis=0, keepdims=True)
        if i > 0:
            y = (a * np.maximum(1 - 1 / (beta * b), 0)).reshape(2 * n)
        else:
            y = np.zeros(2 * n)
        # Solve formula (2.8) from Tao-Yang using the Cholesky factor U
        # (that is, perform a backward solve followed by a forward solve).
        c = D.conj().T @ (y - la / beta) + Ktf
        x = scipy.linalg.lu_solve(U, c)
        # Update the Lagrange multipliers via formula (2.9) from Tao-Yang.
        la = la - beta * (y - D @ x)
    # Calculate the loss in formula (1.4) from Tao-Yang...
    loss = scipy.linalg.norm((D @ x).reshape(2, n), axis=0).sum()
    # ... adding in the term for the fidelity of the reconstruction.
    loss += (mu / 2) * scipy.linalg.norm(K @ x - f)**2
    # Discard the imaginary part of the primal solution,
    # returning only the real part and the loss.
    return x.real, loss


def cs_baseline(m, n, f, mask, mu, beta, n_iter):
    """
    Recovers an image from a subset of its frequencies without any FFTs.

    Reconstructs an m x n image from the subset f of its frequencies specified
    by mask, using ADMM as in function adm (with regularization parameter mu,
    coupling parameter beta, and number of iterations n_iter).
    Unlike function cs_fft, this cs_baseline avoids any FFTs, so runs slowly.

    _N.B._: mask[0, 0] must be True to make the optimization well-posed.

    Parameters
    ----------
    m : int
        number of rows in the image being reconstructed
    n : int
        number of columns in the image being reconstructed
    f : ndarray
        potentially nonzero entries (prior to the inverse Fourier transform)
    mask : numpy.ndarray
        boolean indicators of the potential nonzeros in the full m x n array
        -- note that the zero frequency entry must be True in order to make the
        optimization well-posed
    mu : float
        regularization parameter
    beta : float
        coupling parameter for the ADMM iterations
    n_iter : int
        number of ADMM iterations to conduct

    Returns
    -------
    ndarray
        reconstructed m x n image
    float
        objective value at the end of the ADMM iterations (see function adm)
    """
    assert mask[0, 0]
    # Index the True values in mask.
    mask_nnz = mask.nonzero()
    # Sample the discrete Fourier transform.
    Kx = scipy.linalg.dft(m) / np.sqrt(m)
    Ky = scipy.linalg.dft(n) / np.sqrt(n)
    # Initialize K to be a complex ndarray.
    K = np.zeros((len(mask_nnz[0]), m * n))
    K = K + 0j * K
    # Fill K with the appropriate outer products.
    for k in range(len(mask_nnz[0])):
        outerprod = np.outer(Kx[mask_nnz[0][k], :], Ky[mask_nnz[1][k], :])
        K[k, :] = outerprod.reshape((m * n))
    # Form the forward finite-difference matrices.
    fdx = scipy.linalg.circulant([1, -1] + [0] * (n - 2))
    fdy = scipy.linalg.circulant([1, -1] + [0] * (m - 2))
    # Form the matrix taking finite differences horizontally in an image.
    Dx = np.kron(np.eye(m), fdx)
    # Form the matrix taking finite differences vertically in an image.
    Dy = np.kron(fdy, np.eye(n))
    # Stack the horizontal and vertical finite-difference matrices.
    D = np.vstack((Dx, Dy))
    # Run n_iter iterations of ADMM with coupling beta.
    return adm(f, K, D, mu, beta, n_iter)


def cs_fft(m, n, f, mask, mu, beta, n_iter):
    """
    Recovers an image from a subset of its frequencies using FFTs.

    Reconstructs an m x n image from the subset f of its frequencies specified
    by mask, using ADMM with regularization parameter mu, coupling parameter
    beta, and number of iterations n_iter. Unlike function cs_baseline,
    this cs_fft uses FFTs. The computations take place on the CPU(s) in numpy
    when f is a numpy.ndarray and take place on the GPU(s) in ctorch when f is
    a ctorch.ComplexTensor.

    _N.B._: mask[0, 0] must be True to make the optimization well-posed.

    Parameters
    ----------
    m : int
        number of rows in the image being reconstructed
    n : int
        number of columns in the image being reconstructed
    f : numpy.ndarray or ctorch.ComplexTensor
        potentially nonzero entries (prior to the inverse Fourier transform)
    mask : numpy.ndarray
        boolean indicators of the potential nonzeros in the full m x n array
        -- note that the zero frequency entry must be True in order to make the
        optimization well-posed
    mu : float
        regularization parameter
    beta : float
        coupling parameter for the ADMM iterations
    n_iter : int
        number of ADMM iterations to conduct

    Returns
    -------
    numpy.ndarray or ctorch.ComplexTensor
        reconstructed m x n image
    float
        objective value at the end of the ADMM iterations (see function adm)
    """

    def image_gradient(x):
        """
        First-order finite-differencing both horizontally and vertically.

        Computes a first-order finite-difference approximation to the gradient.

        Parameters
        ----------
        x : numpy.ndarray or ctorch.ComplexTensor
            image (that is, two-dimensional array)

        Returns
        -------
        numpy.ndarray or ctorch.ComplexTensor
            horizontal finite differences of x stacked on top of the vertical
            finite differences (separating horizontal from vertical via the
            initial dimension)
        """
        if isinstance(x, np.ndarray):
            # Wrap the last column of x around to the beginning.
            x_h = np.hstack((x[:, -1:], x))
            # Wrap the last row of x around to the beginning.
            x_v = np.vstack((x[-1:], x))
            # Apply forward differences to the columns of x.
            d_x = (x_h[:, 1:] - x_h[:, :-1])
            # Apply forward differences to the rows of x.
            d_y = (x_v[1:] - x_v[:-1])
            return np.vstack((d_x.ravel(), d_y.ravel()))
        elif isinstance(x, ctorch.ComplexTensor):
            # Wrap the last column of x around to the beginning.
            x_h = ctorch.cat((x[:, -1:], x), dim=1)
            # Wrap the last row of x around to the beginning.
            x_v = ctorch.cat((x[-1:], x), dim=0)
            # Apply forward differences to the columns of x.
            d_x = (x_h[:, 1:] - x_h[:, :-1])
            # Apply forward differences to the rows of x.
            d_y = (x_v[1:] - x_v[:-1])
            return ctorch.cat((d_x, d_y)).view(2, -1)
        else:
            raise TypeError('Input must be a numpy.ndarray ' +
                            'or a ctorch.ComplexTensor.')

    def image_gradient_T(x):
        """
        Transpose of the operator that function image_gradient implements.

        Computes the transpose of the matrix given by function image_gradient.

        Parameters
        ----------
        x : numpy.ndarray or ctorch.ComplexTensor
            stack of two identically shaped arrays

        Returns
        -------
        numpy.ndarray or ctorch.ComplexTensor
            result of applying to x the transpose of function image_gradient
        """
        if isinstance(x, np.ndarray):
            x_h = x[0]
            x_v = x[1]
            # Wrap the first column of x_h around to the end.
            x_h_ext = np.hstack((x_h, x_h[:, :1]))
            # Wrap the first row of x_v around to the end.
            x_v_ext = np.vstack((x_v, x_v[:1]))
            # Apply forward differences to the columns of x.
            d_x = x_h_ext[:, :-1] - x_h_ext[:, 1:]
            # Apply forward differences to the rows of x.
            d_y = x_v_ext[:-1] - x_v_ext[1:]
            return d_x + d_y
        elif isinstance(x, ctorch.ComplexTensor):
            x_h = x[0]
            x_v = x[1]
            # Wrap the first column of x_h around to the end.
            x_h_ext = ctorch.cat((x_h, x_h[:, :1]), dim=1)
            # Wrap the first row of x_v around to the end.
            x_v_ext = ctorch.cat((x_v, x_v[:1]), dim=0)
            # Apply forward differences to the columns of x.
            d_x = x_h_ext[:, :-1] - x_h_ext[:, 1:]
            # Apply forward differences to the rows of x.
            d_y = x_v_ext[:-1] - x_v_ext[1:]
            return d_x + d_y
        else:
            raise TypeError('Input must be a numpy.ndarray ' +
                            'or a ctorch.ComplexTensor.')

    if isinstance(f, np.ndarray):
        assert mask[0, 0]
        # Rescale f and pad with zeros between the mask samples.
        Ktf = (mu / beta) * zero_padded(m, n, f, mask)
        # Calculate the Fourier transform of the convolutional kernels
        # for finite differences.
        tx = np.abs(np.fft.fft([1, -1] + [0] * (m - 2)))**2
        ty = np.abs(np.fft.fft([1, -1] + [0] * (n - 2)))**2
        # Compute the multipliers required to solve formula (2.8) from Tao-Yang
        # in the Fourier domain. The calculation involves broadcasting the
        # Fourier transform of the convolutional kernel for horizontal finite
        # differences over the vertical directions, and broadcasting the
        # Fourier transform of the convolutional kernel for vertical finite
        # differences over the horizontal directions.
        multipliers = 1. / (ty + tx[:, None] + (mu / beta) * mask)
        # Initialize the primal (x) and dual (la) solutions to zeros.
        x = np.zeros((m, n))
        la = np.zeros((2, m * n))
        # Calculate iterations of alternating minimization.
        for i in range(n_iter):
            # Apply shrinkage via formula (2.7) from Tao-Yang, dividing both
            # arguments of the "max" operator in formula (2.7) by the
            # denominator of the rightmost factor in formula (2.7).
            a = image_gradient(x) + la / beta
            b = scipy.linalg.norm(a, axis=0, keepdims=True)
            if i > 0:
                y = a * np.maximum(1 - 1 / (beta * b), 0)
            else:
                y = np.zeros((2, m * n))
            # Solve formula (2.8) from Tao-Yang in the Fourier domain.
            c = image_gradient_T((y - la / beta).reshape((2, m, n))) + Ktf
            x = np.fft.ifft2(np.fft.fft2(c) * multipliers)
            # Update the Lagrange multipliers via formula (2.9) from Tao-Yang.
            la = la - beta * (y - image_gradient(x))
        # Calculate the loss in formula (1.4) from Tao-Yang...
        loss = np.linalg.norm(image_gradient(x), axis=0).sum()
        # ... adding in the term for the fidelity of the reconstruction.
        loss += np.linalg.norm(
            np.fft.fft2(x)[mask] / np.sqrt(m * n) - f)**2 * (mu / 2)
        # Discard the imaginary part of the primal solution,
        # returning only the real part and the loss.
        return x.real, loss
    elif isinstance(f, ctorch.ComplexTensor):
        assert mask[0, 0]
        # Convert the mask from boolean indicators to long integer indices.
        mask_nnz = torch.nonzero(torch.from_numpy(mask.astype(np.uint8)))
        mask_nnz = mask_nnz.cuda()
        # Rescale f and pad with zeros between the mask samples.
        Ktf = zero_padded(m, n, f, mask_nnz) * (mu / beta)
        # Calculate the Fourier transform of the convolutional kernels
        # for finite differences.
        tx = np.abs(np.fft.fft([1, -1] + [0] * (m - 2)))**2
        ty = np.abs(np.fft.fft([1, -1] + [0] * (n - 2)))**2
        # Compute the multipliers required to solve formula (2.8) from Tao-Yang
        # in the Fourier domain. The calculation involves broadcasting the
        # Fourier transform of the convolutional kernel for horizontal finite
        # differences over the vertical directions, and broadcasting the
        # Fourier transform of the convolutional kernel for vertical finite
        # differences over the horizontal directions.
        multipliers = 1. / (ty + tx[:, None] + (mu / beta) * mask)
        multipliers = ctorch.from_numpy(multipliers).cuda()
        # Initialize the primal (x) and dual (la) solutions to zeros,
        # creating new ctorch tensors of the same type as f.
        x = f.new(m, n).zero_()
        la = f.new(2, m * n).zero_()
        # Calculate iterations of alternating minimization.
        for i in range(n_iter):
            # Apply shrinkage via formula (2.7) from Tao-Yang, dividing both
            # arguments of the "max" operator in formula (2.7) by the
            # denominator of the rightmost factor in formula (2.7).
            a = image_gradient(x) + la / beta
            b = ctorch.norm(a, p=2, dim=0, keepdim=True)
            if i > 0:
                y = a * torch.clamp(1 - 1 / (beta * b), min=0)
            else:
                y = f.new(2, m * n).zero_()
            # Solve formula (2.8) from Tao-Yang in the Fourier domain.
            c = image_gradient_T((y - la / beta).view(2, m, n)) + Ktf
            x = ctorch.ifft2(ctorch.fft2(c) * multipliers)
            # Update the Lagrange multipliers via formula (2.9) from Tao-Yang.
            la = la - (y - image_gradient(x)) * beta
        # Calculate the loss in formula (1.4) from Tao-Yang...
        loss = ctorch.norm(image_gradient(x), p=2, dim=0).sum()
        # ... adding in the term for the fidelity of the reconstruction.
        ftx = ctorch.fft2(x) / math.sqrt(m * n)
        mask_flat = mask_nnz[:, 0] * n + mask_nnz[:, 1]
        loss += ctorch.norm(ftx.view(-1)[mask_flat] - f)**2 * (mu / 2)
        # Discard the imaginary part of the primal solution,
        # returning only the real part and the loss.
        return x.real, loss.cpu().item()
    else:
        raise TypeError('Input must be a numpy.ndarray ' +
                        'or a ctorch.ComplexTensor.')


def runtestadmm(method, cpu, filename, mu=1e12, beta=1, subsampling_factor=0.7,
                n_iter=100, seed=None):
    """Run tests as specified.

    Use the specified method (on CPUs if cpu is True), reading the image from
    filename, with the lasso regularization parameter mu and the ADMM coupling
    parameter beta, subsampling by subsampling_factor, for n_iter iterations
    of ADMM, seeding the random number generator with the provided seed.

    Parameters
    ----------
    method : str
        which algorithm to use ('cs_baseline' or 'cs_fft')
    cpu : boolean
        set to true to perform all computations on the CPU(s)
    filename : str
        name of the file containing the image to be processed; prepend a path
        if the file resides outside the working directory
    mu : float
        regularization parameter
    beta : float
        coupling parameter for the ADMM iterations
    subsampling_factor : float
        probability of retaining an entry in k-space
    n_iter : int
        number of ADMM iterations to conduct
    seed : int
        seed value for numpy's random number generator

    Returns
    -------
    float
        objective value at the end of the ADMM iterations (see function adm)
    """

    def tic():
        """
        Timing starting.

        Records the current time.

        Returns
        -------
        float
            present time in fractional seconds
        """
        torch.cuda.synchronize()
        return time.perf_counter()

    def toc(t):
        """
        Timing stopping.

        Reports the difference of the current time from the reference provided.

        Parameters
        ----------
        t : float
            reference time in fractional seconds

        Returns
        -------
        float
            difference of the present time from the reference t
        """
        torch.cuda.synchronize()
        return time.perf_counter() - t

    # Fix the random seed if appropriate.
    np.random.seed(seed=seed)
    # Read the image from disk.
    f_orig = np.array(Image.open(filename)).astype(np.float64) / 255.
    m = f_orig.shape[0]
    n = f_orig.shape[1]
    # Select which k-space frequencies to retain.
    mask = np.random.uniform(size=(m, n)) < subsampling_factor
    # Make the optimization well-posed by including the zero frequency.
    mask[0, 0] = True
    # Subsample the Fourier transform of the original image.
    f = np.fft.fft2(f_orig)[mask] / np.sqrt(m * n)
    # Start timing.
    t = tic()
    # Reconstruct the image from the undersampled Fourier data.
    print('Running {}(cpu={}, mu={}, beta={}, n_iter={})'.format(
        method, cpu, mu, beta, n_iter))
    if method == 'cs_baseline':
        if cpu:
            x, loss = cs_baseline(m, n, f, mask, mu=mu, beta=beta,
                                  n_iter=n_iter)
        else:
            raise NotImplementedError('A baseline on GPUs is not implemented' +
                                      '; use \'--cpu\'')
    elif method == 'cs_fft':
        if cpu:
            x, loss = cs_fft(m, n, f, mask, mu=mu, beta=beta, n_iter=n_iter)
        else:
            # Move the Fourier data to the GPUs.
            f_th = ctorch.from_numpy(f).cuda()
            # The first call to `ctorch.fft2` is slow;
            # run a dummy fft2 and restart the timer to get accurate timings.
            ctorch.fft2(ctorch.from_numpy(np.fft.fft2(f_orig)).cuda())
            t = tic()
            x, loss = cs_fft(m, n, f_th, mask, mu=mu, beta=beta, n_iter=n_iter)
            x = x.cpu().numpy()
    else:
        raise NotImplementedError('method must be either \'cs_baseline\' ' +
                                  'or \'cs_fft\'')
    # Stop timing.
    tt = toc(t)
    # Print the time taken and final loss.
    print('time={}s'.format(tt))
    print('loss={}'.format(loss))
    # Plot the original image, its reconstruction, and the sampling pattern.
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.title('Original')
    plt.imshow(f_orig, cmap='gray')
    plt.subplot(222)
    plt.title('Compressed sensing reconstruction')
    plt.imshow(x.reshape(m, n), cmap='gray')
    plt.subplot(223)
    plt.title('Naive (zero-padded ifft2) reconstruction')
    plt.imshow(np.abs(zero_padded(m, n, f, mask)), cmap='gray')
    plt.subplot(224)
    plt.title('Sampling mask')
    plt.imshow(mask, cmap='gray')
    plt.savefig('recon.png', bbox_inches='tight')
    return loss


if __name__ == '__main__':
    # Parse the command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['cs_baseline', 'cs_fft'],
                        default='cs_fft')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--image', default='data/shepp_logan_32x64.png')
    parser.add_argument('--mu', type=float, default=1e12)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--subsampling_factor', type=float, default=0.7)
    parser.add_argument('--n_iter', type=int, default=100)
    args = parser.parse_args()
    # Run the specified tests.
    runtestadmm(args.method, args.cpu, args.image, args.mu, args.beta,
                args.subsampling_factor, args.n_iter)
