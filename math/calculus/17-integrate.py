#!/usr/bin/env python3
"""
Calculate the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial.

    Args:
        poly (list): List of polynomial coefficients.
        C (int, optional): Integration constant. Defaults to 0.

    Returns:
        list: Coefficients of the integrated polynomial, or None if invalid.
    """
    if (not isinstance(poly, list) or len(poly) == 0 or
            not isinstance(C, (int, float))):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        if val.is_integer():
            val = int(val)
        integral.append(val)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
