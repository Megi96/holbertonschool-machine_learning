#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

# Seed for reproducibility
np.random.seed(0)

# Generate sample Poisson data
data = np.random.poisson(5., 100).tolist()

# Instance from data
p1 = Poisson(data)
print('F(9) (from data):', p1.cdf(9))

# Instance from given lambtha
p2 = Poisson(lambtha=5)
print('F(9) (given lambtha):', p2.cdf(9))
