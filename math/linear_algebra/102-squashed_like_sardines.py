#!/usr/bin/env python3
"""
Concatenate two matrices along a specified axis without using numpy.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a given axis, returning a new matrix.

    Args:
        mat1 (list): First matrix.
        mat2 (list): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        list: New concatenated matrix, or None if shapes mismatch.
    """
    # Base case: axis 0, top-level concatenation
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return None
        return [elem for elem in mat1] + [elem for elem in mat2]

    # Ensure the sublists have the same length for deeper axes
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    # Recursively concatenate each corresponding element
    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis=axis-1)
        if sub_result is None:
            return None
        new_matrix.append(sub_result)

    return new_matrix
