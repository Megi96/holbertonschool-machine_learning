#!/usr/bin/env python3
import numpy as np
BidirectionalCell = __import__('5-bi_forward').BidirectionalCell


class BidirectionalCell(BidirectionalCell):
    """
    Represents a bidirectional cell of an RNN
    """

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction
        for one time step

        Args:
            h_next (numpy.ndarray): shape (m, h)
                next hidden state
            x_t (numpy.ndarray): shape (m, i)
                input at time step t

        Returns:
            h_prev (numpy.ndarray): shape (m, h)
                previous hidden state
        """

        # linear combination of input and next hidden state
        h_prev = np.matmul(x_t, self.Whb) + np.matmul(h_next, self.bhb) + self.bhb

        # apply activation
        h_prev = np.tanh(h_prev)

        return h_prev
