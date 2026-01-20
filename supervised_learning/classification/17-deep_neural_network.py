#!/usr/bin/env python3
"""
17-deep_neural_network.py
Defines a DeepNeuralNetwork class performing binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork.

        nx: number of input features
        layers: list of nodes per layer
        """
        # Input validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)      # number of layers
        self.__cache = {}           # stores activations
        self.__weights = {}         # stores weights and biases

        # Initialize weights and biases using He et al.
        for layer_idx in range(1, self.__L + 1):
            if layer_idx == 1:
                prev_nodes = nx
            else:
                prev_nodes = layers[layer_idx - 2]

            # He initialization: W{layer_idx} ~ N(0, sqrt(2/prev_nodes))
            self.__weights["W" + str(layer_idx)] = (
                np.random.randn(layers[layer_idx - 1], prev_nodes)
                * np.sqrt(2 / prev_nodes)
            )
            self.__weights["b" + str(layer_idx)] = np.zeros(
                (layers[layer_idx - 1], 1)
            )

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights
