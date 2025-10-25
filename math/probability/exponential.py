#!/usr/bin/env python3
"""This module defines an Exponential distribution class."""


class Exponential:
    """
    Represents an Exponential distribution.

    An Exponential distribution describes the time between events in a
    Poisson process, i.e., a process in which events occur continuously
    and independently at a constant average rate.

    Attributes:
        lambtha (float): The expected number of occurrences in a given
            time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

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
            # Lambda for Exponential: 1 / mean
            self.lambtha = float(1 / (sum(data) / len(data)))
