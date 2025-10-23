#!/usr/bin/env python3
"""
Module that calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    Calculate the derivative of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: The derivative of the polynomial.
    """
    # Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    # Compute derivative for valid polynomials
    return [i * poly[i] for i in range(1, len(poly))