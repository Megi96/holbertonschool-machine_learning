#!/usr/bin/env python3
import numpy as np
BidirectionalCell = __import__('6-bi_backward').BidirectionalCell


class BidirectionalCell(BidirectionalCell):
    """
    Bidirectional RNN cell
    """

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Args:
            H (numpy.ndarray): shape (t, m, 2 * h)
                concatenated hidden states (forward + backward)

        Returns:
            Y (numpy.ndarray): shape (t, m, o)
                output probabilities
        """

        t, m, _ = H.shape

        Y = []

        for i in range(t):
            h_t = H[i]

            # linear transformation
            y_linear = np.matmul(h_t, self.Wy) + self.by

            # numerical stability fix
            y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))

            # softmax
            y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

            Y.append(y)

        return np.array(Y)
