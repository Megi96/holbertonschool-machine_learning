#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        nx: number of input features
        layers: list representing number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        # He et al. initialization
        for l, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
            w_key = "W" + str(l + 1)
            b_key = "b" + str(l + 1)
            if l == 0:
                self.weights[w_key] = np.random.randn(nodes, nx) * np.sqrt(2 / nx)
            else:
                self.weights[w_key] = np.random.randn(nodes, layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            self.weights[b_key] = np.zeros((nodes, 1))

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Performs forward propagation
        X: input data
        """
        self.cache["A0"] = X
        for l in range(self.L):
            W = self.weights["W" + str(l + 1)]
            b = self.weights["b" + str(l + 1)]
            A_prev = self.cache["A" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.sigmoid(Z)
            self.cache["A" + str(l + 1)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Calculates cost using logistic regression loss"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the network’s predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent
        """
        m = Y.shape[1]
        L = self.L
        weights_copy = self.weights.copy()
        dZ = cache["A" + str(L)] - Y

        for l in reversed(range(L)):
            A_prev = cache["A" + str(l)]
            W = weights_copy["W" + str(l + 1)]
            self.weights["W" + str(l + 1)] -= alpha * np.dot(dZ, A_prev.T) / m
            self.weights["b" + str(l + 1)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m
            if l > 0:
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network
        X: input data
        Y: true labels
        iterations: number of passes
        alpha: learning rate
        verbose: print cost
        graph: plot cost
        step: steps to print and plot
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not isinstance(step, int) or step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)

            self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
