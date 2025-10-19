#!/usr/bin/env python3
"""
This module defines a function that slices
a NumPy ndarray along specific axes.
"""

import numpy as np


def np_slice(matrix, axes={}):
    """
    Slice a NumPy ndarray along specified axes.

    Args:
        matrix (numpy.ndarray): The input array to slice.
        axes (dict): A dictionary where keys are axes and values
                     are tuples representing the slices to make
                     along that axis.

    Returns:
        numpy.ndarray: The sliced array.
    """
    slices = [slice(*axes.get(i, (None, None, None)))
              for i in range(matrix.ndim)]
    return matrix[tuple(slices)]
