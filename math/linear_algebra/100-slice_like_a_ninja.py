#!/usr/bin/env python3
"""
Slices a matrix along specific axes without importing any module.
"""

def np_slice(matrix, axes={}):
    """
    Slices a matrix (NumPy ndarray) along specified axes.

    Args:
        matrix: The input matrix to slice (a numpy.ndarray).
        axes (dict): A dictionary where each key is an axis (int)
                     and the value is a tuple representing the slice
                     along that axis. Example: {0: (2,), 2: (None, None, -2)}.

    Returns:
        A new numpy.ndarray representing the sliced matrix.
    """
    slices = [slice(None)] * len(matrix.shape)
    for axis, sl in axes.items():
        slices[axis] = slice(*sl)
    return matrix[tuple(slices)]
