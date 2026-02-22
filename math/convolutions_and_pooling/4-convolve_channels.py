#!/usr/bin/env python3
"""Module for performing multi-channel convolution with padding and stride."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multi-channel images with padding and stride.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w, c) containing multiple images.
    kernel : numpy.ndarray
        Array of shape (kh, kw, c) containing the convolution kernel.
    padding : tuple, 'same', or 'valid', optional
        'same'  : output size same as input size.
        'valid' : no padding.
        tuple (ph, pw) : custom padding for height and width.
        Default is 'same'.
    stride : tuple, optional
        Tuple (sh, sw) representing the stride along height and width.
        Default is (1, 1).

    Returns
    -------
    numpy.ndarray
        Array containing the convolved images with shape
        (m, h_out, w_out).
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph_temp = (h - 1) * sh + kh - h
        pw_temp = (w - 1) * sw + kw - w
        ph = ph_temp // 2 + (ph_temp % 2 > 0)
        pw = pw_temp // 2 + (pw_temp % 2 > 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    # Pad images (only height and width)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Compute output dimensions
    h_out = ((h + 2 * ph - kh) // sh) + 1
    w_out = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output
    output = np.zeros((m, h_out, w_out))

    # Perform convolution (2 loops only)
    for i in range(h_out):
        for j in range(w_out):
            patch = images_padded[
                :, i*sh:i*sh + kh, j*sw:j*sw + kw, :
            ]
            # Sum over height, width, and channels
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2, 3))

    return output
