#!/usr/bin/env python3
"""
102-squashed_like_sardines.py

Concatenate two matrices along a specified axis without using numpy.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a given axis.

    Args:
        mat1 (list): First matrix.
        mat2 (list): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        list: Concatenated matrix, or None if the shapes mismatch.
    """
    # Base case: axis 0, simple concatenation if lengths match higher dims
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        return mat1 + mat2

    # Check dimensions match for recursive concatenation
    if len(mat1) != len(mat2):
        return None

    # Recurse along the axis
    result = []
    for m1, m2 in zip(mat1, mat2):
        res = cat_matrices(m1, m2, axis=axis-1)
        if res is None:
            return None
        result.append(res)

    return
