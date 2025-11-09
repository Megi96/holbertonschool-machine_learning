#!/usr/bin/env python3
"""
101-pca.py

Visualizes the Iris dataset using 3D PCA (Principal Component Analysis).

- Reduces 4D data (sepal length, sepal width, petal length, petal width)
  to 3D using PCA.
- Colors points based on species using the 'plasma' colormap.
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Load data, perform PCA, and plot a 3D scatter of Iris dataset."""
    # Load the dataset
    lib = np.load("pca.npz")
    data = lib["data"]      # shape (150, 4)
    labels = lib["labels"]  # shape (150,)

    # Center the data (subtract mean)
    data_means = np.mean(data, axis=0)
    norm_data = data - data_means

    # Compute PCA using Singular Value Decomposition (SVD)
    _, _, Vh = np.linalg.svd(norm_data)
    pca_data = np.matmul(norm_data, Vh[:3].T)  # reduce to 3D

    # Create 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: color points by species using plasma colormap
    ax.scatter(
        pca_data[:, 0],  # U1
        pca_data[:, 1],  # U2
        pca_data[:, 2],  # U3
        c=labels,
        cmap='plasma',
        s=50
    )

    # Label axes
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('U3')

    # Set title
    ax.set_title('PCA of Iris Dataset')

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
