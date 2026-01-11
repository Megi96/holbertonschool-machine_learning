#!/usr/bin/env python3
"""
Defines a NeuralNetwork class with one hidden layer
performing binary classification.
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary classification.

    Attributes
    ----------
    __W1 : numpy.ndarray
        Weights for the hidden layer.
    __b1 : numpy.ndarray
        Biases for the hidden layer.
    __A1 : float or numpy.ndarray
        Activated output for the hidden layer.
    __W2 : numpy.ndarray
        Weights for the output neuron.
    __b2 : float
        Bias for the output neuron.
    __A2 : float or numpy.ndarray
        Activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

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
        """Weights of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Biases of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Weights of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Perform forward propagation of the neural network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m).

        Returns
        -------
        tuple
            Activated outputs (A1, A2) for the hidden layer and output neuron.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Parameters
        ----------
        Y : numpy.ndarray
            Correct labels of shape (1, m).
        A : numpy.ndarray
            Activated output of the output neuron.

        Returns
        -------
        float
            Logistic regression cost.
        """
        m = Y.shape[1]
        return -np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        ) / m

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m).
        Y : numpy.ndarray
            Correct labels of shape (1, m).

        Returns
        -------
        tuple
            The predicted labels (0 or 1) and the cost of the network.
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descent on the network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m).
        Y : numpy.ndarray
            Correct labels of shape (1, m).
        A1 : numpy.ndarray
            Activated output of hidden layer.
        A2 : numpy.ndarray
            Activated output of output neuron.
        alpha : float
            Learning rate.
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neural network over a number of iterations.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (nx, m).
        Y : numpy.ndarray
            Correct labels of shape (1, m).
        iterations : int, optional
            Number of training iterations (default 5000).
        alpha : float, optional
            Learning rate (default 0.05).

        Raises
        ------
        TypeError
            If iterations is not an integer or alpha is not a float.
        ValueError
            If iterations or alpha are not positive.

        Returns
        -------
        tuple
            The predicted labels and cost after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
