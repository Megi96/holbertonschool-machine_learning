#!/usr/bin/env python3
"""
Module that reverses a DataFrame in time and transposes it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order and transposes it.

    Args:
        df (pd.DataFrame): The DataFrame to transform.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Reverse row order
    df_reversed = df.iloc[::-1]
    # Transpose the DataFrame
    return df_reversed.T
