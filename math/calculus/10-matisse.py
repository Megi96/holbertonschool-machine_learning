#!/usr/bin/env python3
"""
Module that contains a function to calculate the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial represented by a list of
    coefficients.

    Args:
        poly (list): List of coefficients where index represents the power
                     of x.

    Returns:
        list: Coefficients of the derivative polynomial.
        None: If poly is not a valid list of numbers.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float))
                                             for c in poly):
        return None

    if len(poly) == 1:
        return [0]

    # Derivative: multiply each coefficient by its power
    derivative = [i * poly[i] for i in range(1, len(poly))]

    return derivative if derivative else [0]
