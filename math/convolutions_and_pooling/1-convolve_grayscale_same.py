#!/usr/bin/env python3
"""Same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)

    Returns: numpy.ndarray of shape (m, h, w)
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Compute padding
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode='constant'
    )

    # Output (same size as input)
    output = np.zeros((m, h, w))

    # Only two loops
    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
