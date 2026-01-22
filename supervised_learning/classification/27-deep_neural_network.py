#!/usr/bin/env python3
"""27-deep_neural_network.py
Defines a deep neural network for multiclass classification
"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network performing multiclass classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list) or len(layers) == 0 or
                not all(isinstance(neurons, int) and neurons > 0
                        for neurons in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(1, self.L + 1):
            prev_size = nx if layer_idx == 1 else layers[layer_idx - 2]
            layer_size = layers[layer_idx - 1]

            self.__weights[f"W{layer_idx}"] = (
                np.random.randn(layer_size, prev_size) *
                np.sqrt(1 / prev_size)
            )
            self.__weights[f"b{layer_idx}"] = np.zeros((layer_size, 1))

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache["A0"] = X

        for layer_idx in range(1, self.L + 1):
            W = self.__weights[f"W{layer_idx}"]
            b = self.__weights[f"b{layer_idx}"]
            A_prev = self.__cache[f"A{layer_idx - 1}"]

            Z = np.matmul(W, A_prev) + b

            if layer_idx != self.L:
                A = np.tanh(Z)
            else:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

            self.__cache[f"A{layer_idx}"] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost using categorical cross-entropy"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-8)) / m

    def evaluate(self, X, Y):
        """Evaluates the neural network"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        dZ = cache[f"A{self.L}"] - Y

        for layer_idx in reversed(range(1, self.L + 1)):
            A_prev = cache[f"A{layer_idx - 1}"]
            W = self.__weights[f"W{layer_idx}"]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_idx > 1:
                dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

            self.__weights[f"W{layer_idx}"] -= alpha * dW
            self.__weights[f"b{layer_idx}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        A_final, _ = self.forward_prop(X)
        cost_final = self.cost(Y, A_final)
        return A_final, cost_final

    def save(self, filename):
        """Saves the instance to a file"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
