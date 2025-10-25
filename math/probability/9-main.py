#!/usr/bin/env python3
"""Test script for Normal CDF"""

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)

# Instance using data
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print('PHI(90):', n1.cdf(90))

# Instance using mean and stddev
n2 = Normal(mean=70, stddev=10)
print('PHI(90):', n2.cdf(90))
