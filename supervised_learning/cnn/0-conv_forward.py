#!/usr/bin/env python3
"""Convolutional forward propagation"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer

    A_prev: (m, h_prev, w_prev, c_prev)
    W: (kh, kw, c_prev, c_new)
    b: (1, 1, 1, c_new)
    activation: activation function
    padding: "same" or "valid"
    stride: (sh, sw)

    Returns:
    Output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # ----- Padding -----
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:  # valid
        ph = 0
        pw = 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    # ----- Output dimensions -----
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # ----- Convolution -----
    for i in range(m):  # loop over examples
        for h in range(h_out):
            for w in range(w_out):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                for c in range(c_new):
                    slice_prev = A_prev_pad[
                        i,
                        vert_start:vert_end,
                        horiz_start:horiz_end,
                        :
                    ]

                    Z[i, h, w, c] = np.sum(
                        slice_prev * W[:, :, :, c]
                    ) + b[:, :, :, c]

    # Apply activation
    return activation(Z)
