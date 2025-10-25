#!/usr/bin/env python3

Poisson = __import__('poisson').Poisson

# Example with data
data = [5, 4, 6, 5, 5, 5, 4, 6, 5, 5]
p1 = Poisson(data)
print('Lambtha (from data):', p1.lambtha)

# Example with given lambtha
p2 = Poisson(lambtha=5)
print('Lambtha (given):', p2.lambtha)

# Example of invalid data (should raise TypeError)
try:
    p3 = Poisson(data="not a list")
except Exception as e:
    print(e)

# Example of too short data (should raise ValueError)
try:
    p4 = Poisson(data=[5])
except Exception as e:
    print(e)

# Example of invalid lambtha (should raise ValueError)
try:
    p5 = Poisson(lambtha=0)
except Exception as e:
    print(e)
