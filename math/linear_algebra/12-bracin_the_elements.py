#!/usr/bin/env python3
"""
This module defines a function that performs
element-wise operations between two matrices.
"""


def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction,
    multiplication, and division between two matrices.

    Args:
        mat1: First matrix, interpretable as a NumPy ndarray.
        mat2: Second matrix, interpretable as a NumPy ndarray.

    Returns:
        tuple: A tuple containing four NumPy ndarrays:
               (sum, difference, product, quotient)
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
