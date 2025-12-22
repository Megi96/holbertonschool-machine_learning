#!/usr/bin/env python3
import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X (numpy.ndarray): shape (n, d), dataset
    k (int): number of clusters

    Returns:
    numpy.ndarray: shape (k, d) initialized centroids
    or None on failure
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if k > n:
        return None

    # Min and max for each dimension
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # Initialize centroids using uniform distribution
    centroids = np.random.uniform(low=mins, high=maxs, size=(k, d))

    return centroids
