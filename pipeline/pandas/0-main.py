#!/usr/bin/env python3
"""
Main file to test the from_numpy function.
"""

import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy

# Generate random NumPy arrays
np.random.seed(0)

A = np.random.randn(5, 8)
print(from_numpy(A))

B = np.random.randn(9, 3)
print(from_numpy(B))
