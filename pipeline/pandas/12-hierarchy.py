#!/usr/bin/env python3
"""
Provides a function to concatenate two pandas DataFrames
(bitstamp and coinbase) into a hierarchical DataFrame
with Timestamp as the first level of the MultiIndex.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Creates a Multindex DataFrame using Timestamp as first level.

    Steps:
        1. Index both DataFrames by 'Timestamp'.
        2. Filter both to include timestamps between 1417411980 and 1417417980.
        3. Concatenate them with MultiIndex keys ['bitstamp', 'coinbase'].
        4. Swap levels so Timestamp is the first level.
        5. Sort the index for chronological order.

    Args:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated hierarchical DataFrame.
    """
    # Index both DataFrames by Timestamp
    df1 = index(df1)
    df2 = index(df2)

    # Filter timestamps
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # Concatenate with source labels
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    # Swap index levels so Timestamp is first
    df = df.swaplevel(0, 1)

    # Sort for chronological order
    df.sort_index(inplace=True)

    return df
