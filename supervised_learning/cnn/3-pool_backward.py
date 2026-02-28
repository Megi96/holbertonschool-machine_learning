#!/usr/bin/env python3
"""Pooling backward propagation"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer

    dA: (m, h_new, w_new, c)
    A_prev: (m, h_prev, w_prev, c)
    kernel_shape: (kh, kw)
    stride: (sh, sw)
    mode: 'max' or 'avg'

    Returns:
    dA_prev
    """

    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):

                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for ch in range(c):

                    if mode == 'max':
                        a_slice = A_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ]

                        mask = (a_slice == np.max(a_slice))

                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ] += mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        da = dA[i, h, w, ch]
                        average = da / (kh * kw)

                        dA_prev[
                            i,
                            vert_start:vert_end,
                            horiz_start:horiz_end,
                            ch
                        ] += np.ones((kh, kw)) * average

    return dA_prev
