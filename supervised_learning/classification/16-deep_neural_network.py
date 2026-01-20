#!/usr/bin/env python3
"""Deep Neural Network performing binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list representing the number of nodes per layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Validate all elements are positive integers using only one loop
        for node_count in layers:
            if not isinstance(node_count, int) or node_count <= 0:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Only loop allowed: initialize weights and biases
        for layer_idx in range(self.L):
            layer_size = layers[layer_idx]
            prev_size = nx if layer_idx == 0 else layers[layer_idx - 1]
            w_key = 'W' + str(layer_idx + 1)
            b_key = 'b' + str(layer_idx + 1)
            self.weights[w_key] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(2 / prev_size)
            )
            self.weights[b_key] = np.zeros((layer_size, 1))
