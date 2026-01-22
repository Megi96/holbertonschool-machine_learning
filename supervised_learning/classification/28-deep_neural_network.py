#!/usr/bin/env python3
"""
DeepNeuralNetwork class for multiclass classification
with support for sigmoid or tanh activations in hidden layers
and softmax output layer.
"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Deep neural network performing multiclass classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        nx [int]: number of input features
        layers [list]: number of nodes per layer
        activation [str]: 'sig' or 'tanh' for hidden layers
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        for l in layers:
            if type(l) is not int or l <= 0:
                raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        prev = nx
        for i, nodes in enumerate(layers, 1):
            self.__weights[f"W{i}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i}"] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache["A0"] = X
        for l in range(1, self.L + 1):
            W = self.weights[f"W{l}"]
            b = self.weights[f"b{l}"]
            A_prev = self.cache[f"A{l - 1}"]

            Z = np.matmul(W, A_prev) + b

            if l != self.L:  # hidden layers
                if self.activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    A = np.tanh(Z)
            else:  # output layer: softmax
                expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = expZ / np.sum(expZ, axis=0, keepdims=True)

            self.__cache[f"A{l}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """Categorical cross-entropy cost"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, _ = self.forward_prop(X)
        prediction = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent"""
        m = Y.shape[1]
        back = {}
        for l in reversed(range(1, self.L + 1)):
            A = cache[f"A{l}"]
            A_prev = cache[f"A{l - 1}"]

            if l == self.L:  # output layer
                dz = A - Y
            else:  # hidden layers
                W_next = self.weights[f"W{l + 1}"]
                dz_next = back[f"dz{l + 1}"]
                if self.activation == 'sig':
                    dz = np.matmul(W_next.T, dz_next) * (A * (1 - A))
                else:  # tanh
                    dz = np.matmul(W_next.T, dz_next) * (1 - A ** 2)

            dW = (1 / m) * np.matmul(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            self.__weights[f"W{l}"] -= alpha * dW
            self.__weights[f"b{l}"] -= alpha * db

            back[f"dz{l}"] = dz

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if (verbose or graph):
            if type(step) is not int or step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        if graph:
            import matplotlib.pyplot as plt
            points = []
            x_points = list(range(0, iterations + 1, step))

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {self.cost(Y, A)}")
            if graph and i % step == 0:
                points.append(self.cost(Y, A))
            self.gradient_descent(Y, cache, alpha)

        # last iteration
        A, _ = self.forward_prop(X)
        if verbose:
            print(f"Cost after {iterations} iterations: {self.cost(Y, A)}")
        if graph:
            points.append(self.cost(Y, A))
            import matplotlib.pyplot as plt
            plt.plot(x_points, points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Save instance as pickle file"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load pickle file"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
