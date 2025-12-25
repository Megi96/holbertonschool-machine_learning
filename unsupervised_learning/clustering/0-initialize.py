#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means algorithm
    """
    # Validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if k > n:
        return None

    # Min and max per dimension
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Initialize centroids (EXACTLY ONE CALL)
    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    return centroids
