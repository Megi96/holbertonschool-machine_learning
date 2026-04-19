#!/usr/bin/env python3
import numpy as np


class BidirectionalCell:
    """
    Bidirectional RNN cell
    """

    def output(self, H):
        """
        Compute outputs from hidden states
        """

        t, m, _ = H.shape

        outputs = []

        for i in range(t):
            h_t = H[i]

            y_linear = np.matmul(h_t, self.Wy) + self.by

            # stable softmax
            exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
            y = exp / np.sum(exp, axis=1, keepdims=True)

            outputs.append(y)

        return np.array(outputs)
