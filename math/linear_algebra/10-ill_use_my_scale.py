#!/usr/bin/env python3
"""
This module defines a function that calculates the shape
of a NumPy ndarray without using loops, conditionals,
or imports.
"""


def np_shape(matrix):
    """
    Calculate the shape of a NumPy ndarray.

    Args:
        matrix: The input NumPy array.

    Returns:
        tuple: A tuple of integers representing the shape of the array.
    """
    return matrix.shape
