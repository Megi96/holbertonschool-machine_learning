#!/usr/bin/env python3
"""Same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Total padding needed
    p_h = kh - 1
    p_w = kw - 1

    # Split padding correctly
    pad_top = p_h // 2
    pad_bottom = p_h - pad_top
    pad_left = p_w // 2
    pad_right = p_w - pad_left

    padded = np.pad(
        images,
        ((0, 0),
         (pad_top, pad_bottom),
         (pad_left, pad_right)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            image_slice = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return output
