#!/usr/bin/env python3
"""Module for performing max and average pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters
    ----------
    images : numpy.ndarray
        Array of shape (m, h, w, c) containing multiple images.
    kernel_shape : tuple
        Tuple (kh, kw) representing the pooling kernel size.
    stride : tuple
        Tuple (sh, sw) representing the stride along height and width.
    mode : str, optional
        'max' for max pooling, 'avg' for average pooling. Default is 'max'.

    Returns
    -------
    numpy.ndarray
        Array containing the pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Compute output dimensions
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, h_out, w_out, c))

    # Perform pooling (2 loops only)
    for i in range(h_out):
        for j in range(w_out):
            patch = images[
                :, i*sh:i*sh + kh, j*sw:j*sw + kw, :
            ]
            if mode == 'max':
                output[:, i, j, :] = np.max(patch, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(patch, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
