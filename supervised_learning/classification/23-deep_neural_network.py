#!/usr/bin/env python3
"""23-deep_neural_network.py"""

import numpy as np
import matplotlib.pyplot as plt
from 22-deep_neural_network import DeepNeuralNetwork


class DeepNeuralNetwork(DeepNeuralNetwork):
    """Deep neural network performing binary classification"""

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network

        Args:
            X (numpy.ndarray): input data of shape (nx, m)
            Y (numpy.ndarray): correct labels of shape (1, m)
            iterations (int): number of iterations to train
            alpha (float): learning rate
            verbose (bool): prints cost if True
            graph (bool): plots cost if True
            step (int): interval for printing/plotting

        Returns:
            tuple: evaluation of the training data (A, cost)
        """

        # ----- VALIDATION -----
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # ----- TRAINING LOOP -----
        costs = []
        steps_list = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps_list.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        # ----- PLOT -----
        if graph:
            plt.plot(steps_list, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
