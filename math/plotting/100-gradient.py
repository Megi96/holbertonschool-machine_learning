#!/usr/bin/env python3
"""This module creates a scatter plot of sampled elevations on a mountain.
The plot displays x and y coordinates (in meters), and uses color to represent
the elevation at each sampled point.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """Displays a scatter plot of sampled elevations on a mountain.

    Each point represents an (x, y) coordinate with its elevation (z)
    shown by color. The function uses a terrain colormap and includes
    axis labels, a title, and a colorbar labeled 'elevation (m)'.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(x, y, c=z, cmap='terrain')
    plt.title('Mountain Elevation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')
    plt.show()
