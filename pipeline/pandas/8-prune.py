#!/usr/bin/env python3
"""
Module that removes rows with NaN values in the 'Close' column.
"""


def prune(df):
    """
    Removes any rows from the DataFrame where
    the 'Close' column has NaN values.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The modified DataFrame with NaN 'Close' values removed.
    """
    return df.dropna(subset=['Close'])
