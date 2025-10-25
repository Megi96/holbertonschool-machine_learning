#!/usr/bin/env python3
"""This module defines a Poisson distribution class."""


class Poisson:
    """
    Represents a Poisson distribution.

    A Poisson distribution describes the probability of a given number
    of events occurring in a fixed interval of time or space.

    Attributes:
        lambtha (float): The expected number of occurrences in a given
            time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution.

        If data is provided, lambtha is estimated from the data. Otherwise,
        the given lambtha is used.

        Args:
            data (list, optional): List of observations to estimate lambtha
            lambtha (float, optional): Expected number of occurrences
                (default 1.)

        Raises:
            TypeError: If data is not a list
            ValueError: If data has fewer than 2 values or if lambtha
                is not a positive value
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, x):
        """
        Compute factorial of x (x!).

        Args:
            x (int): Number to compute factorial of

        Returns:
            int: factorial of x
        """
        if x <= 1:
            return 1
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculate the probability mass function (PMF) for a given number of
        successes k.

        Args:
            k (int or float): Number of successes

        Returns:
            float: PMF value for k
        """
        k = int(k)
        if k < 0:
            return 0

        e_approx = 2.7182818285  # Approximate value of e
        return (
            (self.lambtha ** k)
            * (e_approx ** (-self.lambtha))
            / self.factorial(k)
        )

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

        total = 0
        for i in range(0, k + 1):
            total += self.pmf(i)
        return total
