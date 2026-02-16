#!/usr/bin/env python3
"""Same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # SAME padding (works for odd & even)
    pad_top = (kh - 1) // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = (kw - 1) // 2
    pad_right = kw - 1 - pad_left

    padded = np.pad(
        images,
        ((0, 0),
         (pad_top, pad_bottom),
         (pad_left, pad_right)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    # Only two loops allowed
    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(
                image_slice * kernel,
                axis=(1, 2)
            )

    return output
