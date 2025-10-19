#!/usr/bin/env python3
"""
This module provides a function that calculates the shape
of a NumPy ndarray without using loops or conditionals.
"""

import numpy as np


def np_shape(matrix):
    """
    Calculate the shape of a NumPy ndarray.

    Args:
        matrix (numpy.ndarray): The input NumPy array.

    Returns:
        tuple: A tuple of integers representing the shape of the array.
    """
    return matrix.shape
