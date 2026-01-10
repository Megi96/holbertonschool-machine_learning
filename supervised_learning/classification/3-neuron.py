#!/usr/bin/env python3
"""
Module 3-neuron
Defines a Neuron class with forward propagation and cost calculation
for binary classification
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
        """Getter for the weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)

        Returns:
        numpy.ndarray: activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression

        Parameters:
        Y (numpy.ndarray): correct labels of shape (1, m)
        A (numpy.ndarray): activated output of shape (1, m)

        Returns:
        float: cost of the model
        """
        m = Y.shape[1]
        cost = -np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        ) / m
        return cost
