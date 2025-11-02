#!/usr/bin/env python3
"""
Module to clean, preprocess, and visualize cryptocurrency data.
"""

import pandas as pd
import matplotlib.pyplot as plt

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop 'Weighted_Price' column if present
if 'Weighted_Price' in df.columns:
    df = df.drop(columns=['Weighted_Price'])

# Rename 'Timestamp' to 'Date' and convert to datetime
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index on Date
df.set_index('Date', inplace=True)

# Fill missing Close with previous row value
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing High, Low, Open with the same row's Close
for col in ['High', 'Low', 'Open']:
    df[col].fillna(df['Close'], inplace=True)

# Fill missing volume columns with 0
for col in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[col].fillna(0, inplace=True)

# Filter data from 2017 onward
df = df[df.index >= '2017-01-01']

# Resample to daily intervals and aggregate
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the daily data
df_daily.plot(figsize=(12, 6), title='Daily Cryptocurrency Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
