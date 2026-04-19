#!/usr/bin/env python3
import numpy as np


class BidirectionalCell:
    """
    Bidirectional RNN cell
    """

    def backward(self, h_next, x_t):
        """
        Backward hidden state computation
        """

        h_prev = np.matmul(x_t, self.Whb) + np.matmul(h_next, self.Wbh) + self.bhb
        h_prev = np.tanh(h_prev)

        return h_prev
