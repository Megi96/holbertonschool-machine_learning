#!/usr/bin/env python3
"""
Module 1-neuron
Defines a Neuron class for binary classification
"""

import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance

        Parameters:
        nx (int): number of input features

        Raises:
        TypeError: if nx is not an integer
        ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the weights vector

        Returns:
        numpy.ndarray: the weights of the neuron
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the bias

        Returns:
        float: the bias value
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the activated output

        Returns:
        float: the activation output of the neuron
        """
        return self.__A
