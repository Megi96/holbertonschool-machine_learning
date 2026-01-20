#!/usr/bin/env python3
"""17-deep_neural_network.py
Defines a DeepNeuralNetwork class performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(self.__L):
            layer_size = layers[layer_idx]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if layer_idx == 0 else layers[layer_idx - 1]
            w_key = "W" + str(layer_idx + 1)
            b_key = "b" + str(layer_idx + 1)

            self.__weights[w_key] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights[b_key] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
