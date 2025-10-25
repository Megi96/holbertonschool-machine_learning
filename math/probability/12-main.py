#!/usr/bin/env python3
"""
Test script for the Binomial class.

Generates random data, initializes Binomial instances using data and
parameters, and prints the CDF for a given number of successes.
"""

import numpy as np
Binomial = __import__('binomial').Binomial


def main():
    """Main test function."""
    np.random.seed(0)
    data = np.random.binomial(50, 0.6, 100).tolist()

    # Initialize from data
    b1 = Binomial(data)
    print('F(30):', b1.cdf(30))

    # Initialize from parameters
    b2 = Binomial(n=50, p=0.6)
    print('F(30):', b2.cdf(30))


if __name__ == "__main__":
    main()
