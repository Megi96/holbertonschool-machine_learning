#!/usr/bin/env python3
"""
Test the poly_derivative function.
"""

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))  # Expected output: [3, 0, 3]
