#!/usr/bin/env python3
"""Test the CDF method of the Exponential distribution class."""

import numpy as np
Exponential = __import__('exponential').Exponential

# Seed for reproducibility
np.random.seed(0)

# Generate sample data from an exponential distribution
data = np.random.exponential(0.5, 100).tolist()

# Initialize Exponential instance using data
e1 = Exponential(data)
print('F(1):', e1.cdf(1))

# Initialize Exponential instance using a given lambtha
e2 = Exponential(lambtha=2)
print('F(1):', e2.cdf(1))
