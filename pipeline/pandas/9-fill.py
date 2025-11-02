#!/usr/bin/env python3
"""
Module that cleans a DataFrame by removing 'Weighted_Price'
and filling missing values appropriately.
"""


def fill(df):
    """
    Removes 'Weighted_Price' column and fills missing values:
    - 'Close' with previous row's value.
    - 'High', 'Low', and 'Open' with the same row's 'Close' value.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove 'Weighted_Price' column if it exists
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # Fill missing 'Close' values with the previous row's value
    df['Close'].fillna(method='ffill', inplace=True)

    # Fill missing High, Low, Open with same row's Close
    for col in ['High', 'Low', 'Open']:
        df[col].fillna(df['Close'], inplace=True)

    return df
