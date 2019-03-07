#!/usr/bin/env python3

"""Tabulate pixels in a Cartesian grid that lie on lines through the origin.

This provides a way to sample on a Cartesian grid the points which lie on lines
passing through the origin, choosing the angles of the lines at random.

Functions
---------
randradialines
    Indicates pixels in a Cartesian grid that lie on random radial lines.
"""

import math

import matplotlib
matplotlib.use('agg')
import pylab as plt
import numpy as np


def randradialines(m, n, angles):
    """
    Indicates pixels in a Cartesian grid that lie on random radial lines.

    Generates boolean indicators of pixels in an m x n Cartesian grid that lie
    on any of lin random radial lines through the origin.

    Parameters
    ----------
    m : int
        number of rows in the boolean array being generated
    n : int
        number of columns in the boolean array being generated
    angles : array_like
        angles of radial lines in the boolean array being generated

    Returns
    -------
    numpy.ndarray
        boolean indicators of the pixels that lie on any of the radial lines
    """

    def radialine(angle):
        """
        Coordinates in a Cartesian grid of pixels along a radial line.

        Generates the coordinates of pixels in an m x n Cartesian grid that lie
        along the radial line through the origin with the specified angle.

        Parameters
        ----------
        angle : float
            angle of the radial line

        Returns
        -------
        list
            flattened ordered pairs of coordinates of the pixels which lie on
            the radial line
        """
        c = math.cos(angle)
        s = math.sin(angle)
        scale = 1 / max(abs(c), abs(s)) / max(m, n)
        inds = [0] * (2 * max(m, n) - 2)
        for k in range(0, 2 * max(m, n) - 2):
            r = k / 4.
            # Handle the cosines.
            cr = int(round(m * c * scale * r))
            if c < 0:
                cr = (2 * ((m + 1) // 2) - 1 + cr) % m
            # Handle the sines.
            sr = int(round(n * s * scale * r))
            if s < 0:
                sr = (2 * ((n + 1) // 2) - 1 + sr) % n
            # Flatten the indexing.
            inds[k] = n * cr + sr
        return list(set(inds))

    mask = np.ndarray((m * n), dtype='bool')
    mask[:] = False
    for angle in angles:
        mask[radialine(angle)] = True
    mask = mask.reshape((m, n))
    return mask


if __name__ == '__main__':
    plt.figure(figsize=(12, 12))
    for m in [80, 81]:
        for n in [60, 61]:
            angles = np.random.uniform(low=0, high=(2 * np.pi), size=min(m, n))
            # Plot radial lines with the specified angles on an m x n Cartesian
            # grid of pixels.
            plt.subplot(2, 2, 2 * (m % 2) + n % 2 + 1)
            plt.imshow(np.fft.fftshift(randradialines(m, n, angles)),
                       cmap='gray')
    plt.savefig('mask.png', bbox_inches='tight')
