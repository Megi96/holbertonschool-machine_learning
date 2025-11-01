#!/usr/bin/env python3
"""
Module that fills missing values in the 'Close' column.
"""


def fill(df):
    """
    Fills missing values in the 'Close' column using the previous row's value.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'Close' column.

    Returns:
        pd.DataFrame: The modified DataFrame with filled 'Close' values.
    """
    df['Close'].fillna(method='ffill', inplace=True)
    return df
