#!/usr/bin/env python3
"""This module defines a Normal distribution class."""


class Normal:
    """
    Represents a Normal (Gaussian) distribution.

    A Normal distribution describes a continuous probability distribution
    characterized by a bell-shaped curve.

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
            stddev (float, optional): Standard deviation of the distribution
                (default 1.)

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
