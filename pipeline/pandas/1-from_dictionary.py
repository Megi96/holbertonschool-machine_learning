#!/usr/bin/env python3
"""
Create a Pandas DataFrame from a dictionary.
"""
import pandas as pd
# 1. Define the dictonary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# 2. Create the DataFrame
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
