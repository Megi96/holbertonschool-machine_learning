#!/usr/bin/env python3
"""
Module that sorts a DataFrame by the 'High' column in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by the 'High' column in descending order.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'High' column.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by='High', ascending=False)
