#!/usr/bin/env python3
"""
Test file for from_numpy function.
"""
import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy


np.random.seed(0)  # Set seed for reproducibility

# Test 1: 5x8 array
A = np.random.randn(5, 8)
print(from_numpy(A))

# Test 2: 9x3 array
B = np.random.randn(9, 3)
print(from_numpy(B))
