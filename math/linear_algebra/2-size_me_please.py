#!/usr/bin/env python3
"""
This module contains a function to calculate the shape of a matrix
as a list of integers.
"""


def matrix_shape(matrix):
    """
        Calculate the shape of a nested list (matrix).

        Args:
            matrix (list): A nested list representing the matrix.

        Returns:
            list: A list of integers representing the size of each dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
