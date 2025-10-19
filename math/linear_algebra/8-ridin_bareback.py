#!/usr/bin/env python3
"""
Module 8-ridin_bareback
This module defines a function that performs matrix multiplication
between two 2D matrices.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two 2D matrices element-wise using matrix multiplication rules.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.

    Returns:
        list: A new matrix representing the product, or
        None if the matrices cannot be multiplied.
    """
    # Ensure valid dimensions: columns in mat1 == rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    result = [
        [0 for _ in range(len(mat2[0]))]
        for _ in range(len(mat1))
    ]

    # Perform multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
