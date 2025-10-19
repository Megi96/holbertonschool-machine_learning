#!/usr/bin/env python3
"""
102-squashed_like_sardines.py

This module defines a function to concatenate two matrices along a
specified axis without using numpy for nested lists.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.

    Args:
        mat1 (list): First matrix, can be nested lists.
        mat2 (list): Second matrix, same structure as mat1.
        axis (int): Axis along which to concatenate (default 0).

    Returns:
        list: New matrix after concatenation, or None if shapes mismatch.
    """
    if axis == 0:
        return mat1 + mat2

    # For nested lists, recurse along the axis
    return [
        cat_matrices(m1, m2, axis=axis-1)
        for m1, m2 in zip(mat1, mat2)
    ]
