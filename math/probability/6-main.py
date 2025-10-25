#!/usr/bin/env python3
"""Test the Normal distribution class constructor."""

import numpy as np
Normal = __import__('normal').Normal

# Seed for reproducibility
np.random.seed(0)

# Generate sample data from a normal distribution
data = np.random.normal(70, 10, 100).tolist()

# Initialize Normal instance using data
n1 = Normal(data)
print('Mean:', n1.mean, ', Stddev:', n1.stddev)

# Initialize Normal instance using given mean and stddev
n2 = Normal(mean=70, stddev=10)
print('Mean:', n2.mean, ', Stddev:', n2.stddev)
