#!/usr/bin/env python3
"""Pooling forward propagation"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer

    A_prev: (m, h_prev, w_prev, c_prev)
    kernel_shape: (kh, kw)
    stride: (sh, sw)
    mode: 'max' or 'avg'

    Returns:
    Output of the pooling layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # Initialize output
    A = np.zeros((m, h_out, w_out, c_prev))

    # Perform pooling
    for i in range(m):  # loop over examples
        for h in range(h_out):
            for w in range(w_out):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_prev):
                    slice_prev = A_prev[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        c
                    ]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(slice_prev)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(slice_prev)

    return A
