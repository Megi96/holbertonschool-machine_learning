#!/usr/bin/env python3
"""Test the Normal distribution PDF method."""

import numpy as np
Normal = __import__('normal').Normal

# Seed for reproducibility
np.random.seed(0)

# Generate sample data from a normal distribution
data = np.random.normal(70, 10, 100).tolist()

# Initialize Normal instance using data
n1 = Normal(data)
print('PSI(90):', n1.pdf(90))

# Initialize Normal instance using given mean and stddev
n2 = Normal(mean=70, stddev=10)
print('PSI(90):', n2.pdf(90))
