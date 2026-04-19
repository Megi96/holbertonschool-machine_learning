#!/usr/bin/env python3
import numpy as np


class BidirectionalCell:
    def __init__(self, i, h, o):
        self.i = i
        self.h = h
        self.o = o

        # forward weights
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # backward weights
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # output weights
        self.Wy = np.random.randn(i + 2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        concat = np.concatenate((h_next, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whb) + self.bhb)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def output(self, H):
        """
        H: (t, m, 2h)
        Returns: (t, m, o)
        """
        t, m, _ = H.shape
        Y = np.zeros((t, m, self.o))

        for step in range(t):
            y_linear = np.matmul(H[step], self.Wy) + self.by
            Y[step] = self.softmax(y_linear)

        return Y
