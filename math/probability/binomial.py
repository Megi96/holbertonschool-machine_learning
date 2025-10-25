#!/usr/bin/env python3
"""This module defines a Binomial distribution class."""

class Binomial:
    """
    Represents a Binomial distribution.

    Attributes:
        n (int): Number of Bernoulli trials
        p (float): Probability of success for each trial
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        Args:
            data (list, optional): List of observations to estimate n and p from
            n (int, optional): Number of trials (default 1)
            p (float, optional): Probability of success (default 0.5)

        Raises:
            ValueError: If n <= 0, or p <= 0 or >= 1, or if data has < 2 values
            TypeError: If data is not a list
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p_first = 1 - (variance / mean)
            n_round = mean / p_first

            self.n = round(n_round)
            self.p = mean / self.n
