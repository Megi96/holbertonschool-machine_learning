#!/usr/bin/env python3
"""
Module 6-neuron
Defines a Neuron class with training capability for
binary classification
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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)
        Y (numpy.ndarray): correct labels of shape (1, m)

        Returns:
        tuple: (prediction, cost)
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)
        Y (numpy.ndarray): correct labels of shape (1, m)
        A (numpy.ndarray): activated output of shape (1, m)
        alpha (float): learning rate
        """
        m = Y.shape[1]

        dW = np.matmul(X, (A - Y).T) / m
        db = np.sum(A - Y) / m

        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)
        Y (numpy.ndarray): correct labels of shape (1, m)
        iterations (int): number of iterations to train
        alpha (float): learning rate

        Returns:
        tuple: (prediction, cost) after training
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
