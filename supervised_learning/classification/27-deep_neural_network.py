#!/usr/bin/env python3
"""Deep Neural Network for multiclass classification"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Deep Neural Network class for multiclass classification"""

    def __init__(self, nx, layers):
        """Initialize the network"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i, nodes in enumerate(layers):
            if type(nodes) is not int or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
            key_W = "W{}".format(i + 1)
            key_b = "b{}".format(i + 1)
            if i == 0:
                self.weights[key_W] = np.random.randn(nodes, nx) * np.sqrt(2 / nx)
            else:
                self.weights[key_W] = np.random.randn(nodes, layers[i - 1])
                self.weights[key_W] *= np.sqrt(2 / layers[i - 1])
            self.weights[key_b] = np.zeros((nodes, 1))

    @staticmethod
    def sigmoid(Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def softmax(Z):
        """Softmax activation function"""
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward_prop(self, X):
        """Perform forward propagation"""
        self.cache["A0"] = X
        for i in range(self.L):
            W = self.weights["W{}".format(i + 1)]
            b = self.weights["b{}".format(i + 1)]
            A_prev = self.cache["A{}".format(i)]
            Z = np.matmul(W, A_prev) + b
            if i == self.L - 1:  # last layer = softmax
                self.cache["A{}".format(i + 1)] = self.softmax(Z)
            else:
                self.cache["A{}".format(i + 1)] = self.sigmoid(Z)
        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """Compute cost using cross-entropy"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """Evaluate network predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent"""
        m = Y.shape[1]
        dZ = cache["A{}".format(self.L)] - Y
        for i in reversed(range(self.L)):
            A_prev = cache["A{}".format(i)]
            W = self.weights["W{}".format(i + 1)]
            self.weights["W{}".format(i + 1)] -= alpha * np.matmul(dZ, A_prev.T) / m
            self.weights["b{}".format(i + 1)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                A_prev = cache["A{}".format(i)]
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=1):
        """Train the deep neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
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

    def save(self, filename):
        """Save the instance to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork instance"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
