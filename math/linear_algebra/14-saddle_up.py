#!/usr/bin/env python3
"""
This module defines a function that performs
matrix multiplication using NumPy.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication between two NumPy arrays.

    Args:
        mat1: First matrix (numpy.ndarray).
        mat2: Second matrix (numpy.ndarray).

    Returns:
        numpy.ndarray: The result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
