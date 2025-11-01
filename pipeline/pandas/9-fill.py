#!/usr/bin/env python3
"""
Module that cleans a DataFrame by removing 'Weighted_Price'
and filling missing 'Close' values.
"""

def fill(df):
    """
    Removes 'Weighted_Price' column and fills missing 'Close' values
    with the previous row's value.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove 'Weighted_Price' column
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # Fill missing 'Close' values with the previous row's value
    df['Close'].fillna(method='ffill', inplace=True)

    return df
