#!/usr/bin/env python3
"""Deep Neural Network for binary classification"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Deep Neural Network class for binary classification"""

    def __init__(self, nx, layers):
        """Initialize the network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx, nodes in enumerate(layers):
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
            key_W = "W{}".format(layer_idx + 1)
            key_b = "b{}".format(layer_idx + 1)
            if layer_idx == 0:
                self.__weights[key_W] = np.random.randn(nodes, nx) * np.sqrt(2 / nx)
            else:
                self.__weights[key_W] = np.random.randn(nodes, layers[layer_idx - 1])
                self.__weights[key_W] *= np.sqrt(2 / layers[layer_idx - 1])
            self.__weights[key_b] = np.zeros((nodes, 1))

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Forward propagation through all layers"""
        self.__cache["A0"] = X
        for layer_idx in range(self.L):
            W = self.__weights["W{}".format(layer_idx + 1)]
            b = self.__weights["b{}".format(layer_idx + 1)]
            A_prev = self.__cache["A{}".format(layer_idx)]
            Z = np.dot(W, A_prev) + b
            self.__cache["A{}".format(layer_idx + 1)] = self.sigmoid(Z)
        return self.__cache["A{}".format(self.L)], self.__cache

    def cost(self, Y, A):
        """Compute cost using logistic regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) *
                               np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate network predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent over the network"""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.L)] - Y

        for layer_idx in reversed(range(self.L)):
            A_prev = cache["A{}".format(layer_idx)]
            W = self.__weights["W{}".format(layer_idx + 1)]
            self.__weights["W{}".format(layer_idx + 1)] -= alpha * np.dot(
                dZ, A_prev.T) / m
            self.__weights["b{}".format(layer_idx + 1)] -= alpha * np.sum(
                dZ, axis=1, keepdims=True) / m
            if layer_idx > 0:
                A_prev = cache["A{}".format(layer_idx)]
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not isinstance(step, int) or step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                costs.append(cost)
                steps.append(i)
            self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
