#!/usr/bin/env python3
"""
Module 7-gettin_cozy
This module defines a function that concatenates two 2D matrices
along a given axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.
        axis (int): The axis to concatenate along.
                    0 -> vertically, 1 -> horizontally.

    Returns:
        list: A new matrix resulting from concatenation, or
        None if the matrices cannot be concatenated.
    """
    # Validate axis and dimensions
    if axis == 0:
        # Same number of columns
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        # Same number of rows
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    return None
