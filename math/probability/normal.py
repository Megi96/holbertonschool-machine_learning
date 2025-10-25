#!/usr/bin/env python3
"""This module defines a Normal (Gaussian) distribution class."""


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
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.

        Args:
            x (float): The x-value.

        Returns:
            float: z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value for a given z-score.

        Args:
            z (float): The z-score.

        Returns:
            float: x-value corresponding to z.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculate the probability density function (PDF) for a given x.

        Args:
            x (float): The x-value.

        Returns:
            float: PDF value for x.
        """
        e = 2.7182818285
        pi = 3.1415926536
        return (1 / (self.stddev * (2 * pi) ** 0.5) *
                e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        Calculate the cumulative distribution function (CDF) for x.

        Args:
            x (float): The x-value.

        Returns:
            float: CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        # Approximation using erf formula
        t = z
        # Maclaurin series approximation for erf(z)
        erf = (2 / 3.1415926536 ** 0.5) * (t - (t ** 3) / 3 + (t ** 5) / 10 -
                                           (t ** 7) / 42 + (t ** 9) / 216)
        return 0.5 * (1 + erf)
