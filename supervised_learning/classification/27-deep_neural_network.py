#!/usr/bin/env python3
"""27-deep_neural_network.py"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list)) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(type(l) != int or l <= 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He initialization
        for i in range(self.L):
            layer_size = layers[i]
            prev_size = nx if i == 0 else layers[i - 1]
            self.weights['W' + str(i + 1)] = np.random.randn(
                layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.weights['b' + str(i + 1)] = np.zeros((layer_size, 1))

    def forward_prop(self, X):
        """Perform forward propagation."""
        self.cache['A0'] = X
        for l in range(1, self.L + 1):
            W = self.weights['W' + str(l)]
            b = self.weights['b' + str(l)]
            A_prev = self.cache['A' + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid activation
            self.cache['A' + str(l)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Compute cost using logistic regression loss."""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions."""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, alpha=0.05):
        """Perform one pass of gradient descent."""
        m = Y.shape[1]
        L = self.L
        dA = self.cache['A' + str(L)] - Y

        for l in reversed(range(1, L + 1)):
            A_prev = self.cache['A' + str(l - 1)]
            W = self.weights['W' + str(l)]
            dW = np.dot(dA, A_prev.T) / m
            db = np.sum(dA, axis=1, keepdims=True) / m
            self.weights['W' + str(l)] -= alpha * dW
            self.weights['b' + str(l)] -= alpha * db
            if l > 1:
                A_prev_val = self.cache['A' + str(l - 1)]
                dA = np.dot(W.T, dA) * (A_prev_val * (1 - A_prev_val))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the network."""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)
            if (verbose and i % step == 0) or i == iterations:
                print(f"Cost after {i} iterations: {cost}")
            if graph and (i % step == 0 or i == iterations):
                costs.append(cost)
                steps.append(i)
            if i < iterations:
                self.gradient_descent(Y, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
