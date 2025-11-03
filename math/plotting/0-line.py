#!/usr/bin/env python3
"""
A script that plots y as a solid red line.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot y = x³ as a solid red line with x ranging from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='r', linestyle='-')
    plt.xlim(0, 10)
    plt.show()
