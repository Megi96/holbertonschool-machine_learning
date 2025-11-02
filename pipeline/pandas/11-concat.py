#!/usr/bin/env python3
"""
Module that concatenates two DataFrames (bitstamp and coinbase)
after filtering and indexing by 'Timestamp'.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames with multi-index labels.

    Steps:
        1. Index both DataFrames by 'Timestamp'.
        2. Keep only rows in df2 (bitstamp) where Timestamp <= 1417411920.
        3. Concatenate df2 (filtered) on top of df1 (coinbase).
        4. Add keys to distinguish between the two sources.

    Args:
        df1 (pd.DataFrame): The coinbase DataFrame.
        df2 (pd.DataFrame): The bitstamp DataFrame.

    Returns:
        pd.DataFrame: A concatenated DataFrame with labeled sources.
    """
    # Ensure both DataFrames are indexed by 'Timestamp'
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2 up to and including timestamp 1417411920
    df2 = df2.loc[:1417411920]

    # Concatenate and label the sources
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return df
