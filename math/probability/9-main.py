#!/usr/bin/env python3
"""Test the Normal distribution CDF method."""

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)

# Sample data
data = np.random.normal(70, 10, 100).tolist()

# Initialize Normal instance from data
n1 = Normal(data)
print('PHI(90):', n1.cdf(90))

# Initialize Normal instance from given mean and stddev
n2 = Normal(mean=70, stddev=10)
print('PHI(90):', n2.cdf(90))
