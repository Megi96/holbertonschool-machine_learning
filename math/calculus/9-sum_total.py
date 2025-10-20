#!/usr/bin/env python3
"""
Module that contains a function to calculate the sum of squares of integers
from 1 to n.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n using the formula
    n(n+1)(2n+1)/6.

    Args:
        n (int): The stopping integer (must be a positive integer).

    Returns:
        int: Sum of squares from 1 to n.
        None: If n is not a valid positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    return n * (n + 1) * (2 * n + 1) // 6
