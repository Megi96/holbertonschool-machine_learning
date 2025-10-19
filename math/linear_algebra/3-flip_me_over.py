#!/usr/bin/env python3
"""
This module contains a function that returns
the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The input 2D matrix.

    Returns:
        list of lists: A new matrix that is the transpose of the input.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
