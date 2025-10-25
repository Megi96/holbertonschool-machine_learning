#!/usr/bin/env python3
"""Test the Normal distribution z-score and x-value methods."""

import numpy as np
Normal = __import__('normal').Normal

# Seed for reproducibility
np.random.seed(0)

# Generate sample data from a normal distribution
data = np.random.normal(70, 10, 100).tolist()

# Initialize Normal instance using data
n1 = Normal(data)
print('Z(90):', n1.z_score(90))
print('X(2):', n1.x_value(2))

# Initialize Normal instance using given mean and stddev
n2 = Normal(mean=70, stddev=10)
print()
print('Z(90):', n2.z_score(90))
print('X(2):', n2.x_value(2))
