#!/usr/bin/env python3
"""
This module defines a function that concatenates
two matrices along a specified axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a given axis.

    Args:
        mat1: First matrix, interpretable as a NumPy ndarray.
        mat2: Second matrix, interpretable as a NumPy ndarray.
        axis: Axis along which the matrices will be concatenated (default: 0).

    Returns:
        numpy.ndarray: A new concatenated NumPy array.
    """
    return np.concatenate((mat1, mat2), axis=axis)
