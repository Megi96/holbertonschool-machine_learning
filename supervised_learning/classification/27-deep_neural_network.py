#!/usr/bin/env python3
"""27-deep_neural_network.py
DeepNeuralNetwork class for multiclass classification
"""
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network performing multiclass classification"""

    def __init__(self, nx, layers):
        """Initialize the network"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or \
           not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization
        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]
            self.__weights['W' + str(l)] = (np.random.randn(layer_size, prev_size) *
                                            np.sqrt(2 / prev_size))
            self.__weights['b' + str(l)] = np.zeros((layer_size, 1))

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation with sigmoid for hidden, softmax for output"""
        self.__cache['A0'] = X
        for l in range(1, self.L + 1):
            Wl = self.__weights['W' + str(l)]
            bl = self.__weights['b' + str(l)]
            Al_prev = self.__cache['A' + str(l - 1)]
            Zl = np.dot(Wl, Al_prev) + bl

            if l != self.L:  # Hidden layers: sigmoid
                Al = 1 / (1 + np.exp(-Zl))
            else:  # Output layer: softmax
                t = np.exp(Zl - np.max(Zl, axis=0, keepdims=True))
                Al = t / np.sum(t, axis=0, keepdims=True)
            self.__cache['A' + str(l)] = Al
        return Al, self.__cache

    def cost(self, Y, A):
        """Cross-entropy cost for multiclass"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions and cost"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return A, cost if Y.shape[0] > 1 else predictions

    def gradient_descent(self, Y, cache, alpha=0.05):
        """One pass of gradient descent"""
        m = Y.shape[1]
        dZ = cache['A' + str(self.L)] - Y  # Softmax derivative

        for l in reversed(range(1, self.L + 1)):
            Al_prev = cache['A' + str(l - 1)]
            Wl = self.__weights['W' + str(l)]

            dW = np.dot(dZ, Al_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if l > 1:
                Al_prev_sigmoid = Al_prev
                dZ = np.dot(Wl.T, dZ) * (Al_prev_sigmoid * (1 - Al_prev_sigmoid))

            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=False, step=10):
        """Train the network (multiclass aware)"""
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
        if not isinstance(step, int) or step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and (i % step == 0 or i == iterations):
                print(f"Cost after {i} iterations: {cost}")
            if graph:
                costs.append(cost)

            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(0, iterations + 1, step), costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        A_final, _ = self.forward_prop(X)
        cost_final = self.cost(Y, A_final)
        return A_final, cost_final

    def save(self, filename):
        """Save the object in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
