#!/usr/bin/env python3
"""
Module that cleans a DataFrame by removing 'Weighted_Price'
and filling missing values appropriately.
"""


def fill(df):
    """
    Cleans the DataFrame by:
    - Removing 'Weighted_Price'
    - Filling missing 'Close' with previous row's value
    - Filling missing 'High', 'Low', and 'Open' with same row's 'Close' value
    - Filling missing 'Volume_(BTC)' and 'Volume_(Currency)' with 0

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove 'Weighted_Price' column if it exists
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # Fill missing 'Close' with previous row's value
    df['Close'].fillna(method='ffill', inplace=True)

    # Fill missing High, Low, Open with same row's Close
    for col in ['High', 'Low', 'Open']:
        df[col].fillna(df['Close'], inplace=True)

    # Fill missing Volume_(BTC) and Volume_(Currency) with 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col].fillna(0, inplace=True)

    return df
