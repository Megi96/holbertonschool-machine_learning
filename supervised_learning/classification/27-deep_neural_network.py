#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification with persistence"""

import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep neural network for multiclass classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W1"] = (
                    np.random.randn(layers[i], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                self.__weights["W{}".format(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1])
                )
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation using sigmoid for hidden, softmax for output"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]
            Z = np.matmul(W, A_prev) + b

            # Output layer uses softmax
            if i == self.__L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))  # Sigmoid for hidden layers

            self.__cache["A{}".format(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Categorical cross-entropy cost for multiclass"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions for multiclass"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent using one-hot labels and softmax at output"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache["A{}".format(L)] - Y

        for i in reversed(range(1, L + 1)):
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                A_prev_sig = cache["A{}".format(i - 1)]
                dZ = np.matmul(W.T, dZ) * (A_prev_sig * (1 - A_prev_sig))

            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=False):
        """Trains the network (multiclass aware)"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")

        costs = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and (i % 10 == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            if graph:
                costs.append(cost)

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations + 1), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
