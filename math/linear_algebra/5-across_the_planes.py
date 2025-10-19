#!/usr/bin/env python3
"""Module that defines a function to add two 2D matrices element-wise."""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
        mat1 (list of lists): First matrix of integers or floats.
        mat2 (list of lists): Second matrix of integers or floats.

    Returns:
        list of lists: A new matrix with element-wise sums.
        None: If the matrices are not the same shape.
    """
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
