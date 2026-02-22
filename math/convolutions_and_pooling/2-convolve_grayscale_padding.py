#!/usr/bin/env python3
"""Module for performing grayscale convolution with custom padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w) containing multiple grayscale images.
        m is the number of images, h is the height, w is the width.
    kernel : numpy.ndarray
        Array of shape (kh, kw) containing the kernel for the convolution.
        kh is the height of the kernel, kw is the width of the kernel.
    padding : tuple
        Tuple of (ph, pw) representing the padding for height and width.

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images with shape
        (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images with zeros
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Calculate output dimensions
    h_out = h + 2 * ph - kh + 1
    w_out = w + 2 * pw - kw + 1

    # Initialize output array
    output = np.zeros((m, h_out, w_out))

    # Perform convolution (2 loops over height and width)
    for i in range(h_out):
        for j in range(w_out):
            # Extract patch and perform element-wise multiplication and sum
            patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
