#!/usr/bin/env python3

import numpy as np
Exponential = __import__('exponential').Exponential

# Seed for reproducibility
np.random.seed(0)

# Generate sample exponential data
data = np.random.exponential(0.5, 100).tolist()

# Instance from data
e1 = Exponential(data)
print('Lambtha (from data):', e1.lambtha)

# Instance from given lambtha
e2 = Exponential(lambtha=2)
print('Lambtha (given lambtha):', e2.lambtha)
