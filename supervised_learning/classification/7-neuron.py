#!/usr/bin/env python3
"""
Defines a Neuron class for binary classification using logistic regression
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize the neuron

        Parameters:
        nx (int): number of input features

        Raises:
        TypeError: if nx is not an integer
        ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
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
        Computes the logistic regression cost

        Parameters:
        Y (numpy.ndarray): correct labels (1, m)
        A (numpy.ndarray): activated output (1, m)

        Returns:
        float: cost
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
        X (numpy.ndarray): input data (nx, m)
        Y (numpy.ndarray): correct labels (1, m)

        Returns:
        tuple: (prediction, cost)
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent

        Parameters:
        X (numpy.ndarray): input data
        Y (numpy.ndarray): correct labels
        A (numpy.ndarray): activated output
        alpha (float): learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(
        self,
        X,
        Y,
        iterations=5000,
        alpha=0.05,
        verbose=True,
        graph=True,
        step=100
    ):
        """
        Trains the neuron

        Parameters:
        X (numpy.ndarray): input data
        Y (numpy.ndarray): correct labels
        iterations (int): number of training iterations
        alpha (float): learning rate
        verbose (bool): print cost during training
        graph (bool): plot cost after training
        step (int): interval for printing and plotting

        Returns:
        tuple: (prediction, cost)
        """

        # ----- Parameter validation -----
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        # ----- Training loop -----
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        # ----- Plot training cost -----
        if graph:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
