#!/usr/bin/env python3
"""
Defines a NeuralNetwork class with one hidden layer
performing binary classification.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer.
    """

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance.

        Parameters
        ----------
        nx : int
            Number of input features.
        nodes : int
            Number of nodes in the hidden layer.

        Raises
        ------
        TypeError
            If nx or nodes is not an integer.
        ValueError
            If nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get the weights of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Get the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Get the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Get the weights of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Get the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Get the activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculate forward propagation of the neural network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m).

        Returns
        -------
        tuple
            The activated outputs (A1, A2).
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
