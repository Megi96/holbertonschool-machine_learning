#!/usr/bin/env python3
"""
A script that plots the exponential decay of two radioactive elements.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plot exponential decay of C-14 and Ra-226 on the same graph.
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730  # half-life of C-14
    t2 = 1600  # half-life of Ra-226
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, 'r--', label='C-14')   # dashed red
    plt.plot(x, y2, 'g-', label='Ra-226')  # solid green

    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.show()
