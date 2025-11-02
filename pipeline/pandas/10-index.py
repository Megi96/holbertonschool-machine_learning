#!/usr/bin/env python3
"""
Module that sets the 'Timestamp' column as the DataFrame index.
"""


def index(df):
    """
    Set the 'Timestamp' column as the index of the DataFrame.

    If the 'Timestamp' column is present, it will become the index.
    The function returns the modified DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame that contains a 'Timestamp' column.

    Returns:
        pd.DataFrame: The DataFrame with 'Timestamp' as its index.
    """
    if 'Timestamp' in df.columns:
        return df.set_index('Timestamp')
    return df
