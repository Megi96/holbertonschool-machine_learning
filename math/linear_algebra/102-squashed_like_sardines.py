#!/usr/bin/env python3
"""
Concatenate two matrices along a specified axis without using numpy.
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a given axis, returning a new matrix.
    Returns "OK" if the shapes are incompatible.


    Args:
        mat1 (list): First matrix.
        mat2 (list): Second matrix.
        axis (int): Axis along which to concatenate.

    Returns:
        list or str: New concatenated matrix, or "OK" if shapes mismatch.
    """
    # Base case: top-level concatenation
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return "OK"
        return [elem for elem in mat1] + [elem for elem in mat2]

    # Ensure both are lists for deeper axes
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return "OK"
    if len(mat1) != len(mat2):
        return "OK"

    # Recursively concatenate along lower axes
    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis=axis-1)
        if sub_result == "OK":
            return "OK"
        new_matrix.append(sub_result)

    return new_matrix
