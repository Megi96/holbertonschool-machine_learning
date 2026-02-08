#!/usr/bin/env python3
"""
13-batch_norm
Performs batch normalization on an unactivated output of a neural network.
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output Z using batch normalization.

    Args:
        Z (numpy.ndarray): Shape (m, n), unactivated outputs
            m = number of data points
            n = number of features
        gamma (numpy.ndarray): Shape (1, n), scale parameters
        beta (numpy.ndarray): Shape (1, n), offset parameters
        epsilon (float): Small number to avoid division by zero

    Returns:
        numpy.ndarray: Normalized Z of shape (m, n)
    """
    # Compute mean of each feature
    mu = np.mean(Z, axis=0, keepdims=True)
    # Compute variance of each feature
    var = np.var(Z, axis=0, keepdims=True)
    # Normalize
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    # Scale and shift
    Z_scaled = gamma * Z_norm + beta

    return Z_scaled
