#!/usr/bin/env python3
"""This module defines a Normal distribution class."""


class Normal:
    """
    Represents a Normal (Gaussian) distribution.

    Attributes:
        mean (float): The mean (average) of the distribution.
        stddev (float): The standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        If data is provided, mean and stddev are estimated from the data.
        Otherwise, the given mean and stddev are used.

        Args:
            data (list, optional): List of observations to estimate mean
                and stddev.
            mean (float, optional): Mean of the distribution (default 0.)
            stddev (float, optional): Standard deviation (default 1.)

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than 2 values or stddev is not
                positive.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate the z-score of a given x-value."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value of a given z-score."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate the probability density function (PDF) for x."""
        e = 2.7182818285
        pi = 3.1415926536
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * (e ** exponent)

    def cdf(self, x):
        """
        Calculate the cumulative distribution function (CDF) for x
        using an approximation of the error function.
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        t = 1.0 / (1.0 + 0.3275911 * abs(z))
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        erf_approx = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
        erf_approx *= 2.7182818285 ** (-z ** 2)
        if z >= 0:
            return 0.5 * (1 + (1 - erf_approx))
        return 0.5 * (1 - (1 - erf_approx))
