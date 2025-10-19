#!/usr/bin/env python3
"""Concatenate two matrices along a specified axis without using numpy."""


def cat_matrices(mat1, mat2, axis=0):
    """Recursively concatenate two matrices along a specified axis.

    Args:
        mat1 (list): First matrix (nested lists).
        mat2 (list): Second matrix (nested lists).
        axis (int): Axis along which to concatenate.

    Returns:
        list: New concatenated matrix or None if shapes mismatch.
    """
    if axis == 0:
        return mat1 + mat2
    try:
        return [
            cat_matrices(m1, m2, axis=axis - 1)
            for m1, m2 in zip(mat1, mat2)
        ]
    except Exception:
        return None
