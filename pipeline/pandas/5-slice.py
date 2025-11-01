#!/usr/bin/env python3
"""
Module that slices a DataFrame by selecting certain columns and every 60th row.
"""


def slice(df):
    """
    Extracts 'High', 'Low', 'Close', and 'Volume_(BTC)' columns and selects
    every 60th row.

    Args:
        df (pd.DataFrame): The DataFrame containing the required columns.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
