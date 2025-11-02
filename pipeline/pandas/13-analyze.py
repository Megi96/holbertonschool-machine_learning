#!/usr/bin/env python3
"""
Module that computes descriptive statistics for a DataFrame,
excluding the Timestamp column.
"""

import pandas as pd


def analyze(df):
    """
    Computes descriptive statistics for all columns except 'Timestamp'.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics
        (count, mean, std, min, 25%, 50%, 75%, max) for each column
        except 'Timestamp'.
    """
    # Exclude Timestamp column
    df_numeric = df.drop(columns=['Timestamp'], errors='ignore')

    # Compute descriptive statistics
    stats = df_numeric.describe()

    return stats
