#!/usr/bin/env python3
"""
A module that creates a Pandas DataFrame from a NumPy array.
"""
import pandas as pd


def from_numpy(array):
    """
    Create a DataFrame from a NumPy array.

    Args:
        array (np.ndarray): The NumPy array to convert.

    Returns:
        pd.DataFrame: DataFrame with columns labeled A, B, C, ...
    """
    if not hasattr(array, 'shape'):
        return None

    # Generate column labels (A, B, C, ...)
    columns = [chr(65 + i) for i in range(array.shape[1])]

    return pd.DataFrame(array, columns=columns)
