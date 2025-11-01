#!/usr/bin/env python3
"""
Module that renames and converts timestamp column in a DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime' and converts its
    values to datetime objects. Displays only the 'Datetime' and
    'Close' columns.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Datetime'
        and 'Close' columns.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    return df[['Datetime', 'Close']]
