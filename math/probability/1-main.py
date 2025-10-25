#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

# Seed for reproducibility
np.random.seed(0)

# Generate sample Poisson data
data = np.random.poisson(5., 100).tolist()

# Instance from data
p1 = Poisson(data)
print('P(9) (from data):', p1.pmf(9))

# Instance from given lambtha
p2 = Poisson(lambtha=5)
print('P(9) (given lambtha):', p2.pmf(9))

# Optional: testing invalid k values
p3 = Poisson(lambtha=3)
print('P(-1):', p3.pmf(-1))  # Should print 0
