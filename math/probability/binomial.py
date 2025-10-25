#!/usr/bin/env python3
"""This module defines a Binomial distribution class."""


class Binomial:
    """
    Represents a Binomial distribution.

    A Binomial distribution describes the number of successes in a fixed
    number of independent Bernoulli trials with the same probability of
    success.

    Attributes:
        n (int): Number of Bernoulli trials
        p (float): Probability of success for each trial
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        If data is provided, n and p are estimated from the data. Otherwise,
        n and p are taken from the arguments.

        Args:
            data (list, optional): List of observations to estimate n and p
                from
            n (int, optional): Number of trials (default 1)
            p (float, optional): Probability of success (default 0.5)

        Raises:
            TypeError: If data is not a list
            ValueError: If n <= 0, p <= 0 or >= 1, or if data has fewer than
                2 values
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

    def pmf(self, k):
        """
        Calculate the probability mass function (PMF) for a given number of
        successes k.

        Args:
            k (int or float): Number of successes

        Returns:
            float: PMF value for k
        """
        from math import comb

        k = int(k)
        if k < 0 or k > self.n:
            return 0

        return comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculate the cumulative distribution function (CDF) for a given
        number of successes k.

        Args:
            k (int or float): Number of successes

        Returns:
            float: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        return sum(self.pmf(i) for i in range(0, k + 1))
