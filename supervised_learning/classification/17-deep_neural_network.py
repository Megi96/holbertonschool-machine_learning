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
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers is a non-empty list
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)      # number of layers
        self.__cache = {}           # stores activations
        self.__weights = {}         # stores weights and biases

        # Single loop: validate each layer and initialize weights/biases
        for layer_idx in range(self.__L):
            layer_size = layers[layer_idx]

            # Validate layer size
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Previous layer size (nx for first layer)
            prev_size = nx if layer_idx == 0 else layers[layer_idx - 1]

            # Keys for weights and biases
            w_key = "W" + str(layer_idx + 1)
            b_key = "b" + str(layer_idx + 1)

            # He initialization for weights
            self.__weights[w_key] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            # Biases initialized to zeros
            self.__weights[b_key] = np.zeros((layer_size, 1))

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
