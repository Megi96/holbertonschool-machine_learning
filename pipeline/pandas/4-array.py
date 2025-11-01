#!/usr/bin/env python3
"""
Module that converts selected DataFrame rows into a NumPy array.
"""

import numpy as np


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    from the given DataFrame and converts them to a NumPy array.

    Args:
        df (pd.DataFrame): The DataFrame containing 'High' and 'Close' columns.

    Returns:
        np.ndarray: A NumPy array with the selected values.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
