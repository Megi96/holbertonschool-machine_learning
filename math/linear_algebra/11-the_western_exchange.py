#!/usr/bin/env python3
"""
This module defines a function that transposes a matrix
interpretable as a NumPy ndarray, without using loops,
conditionals, or imports.
"""


def np_transpose(matrix):
    """
    Return the transpose of a NumPy ndarray.

    Args:
        matrix: The input matrix, which can be interpreted
                as a NumPy ndarray.

    Returns:
        numpy.ndarray: A new array that is the transpose
                       of the input matrix.
    """
    return matrix.T
