#!/usr/bin/env python3
"""
4-moving_average
Calculates the bias-corrected moving average of a dataset
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set using
    bias correction.

    Args:
        data (list): List of numerical values
        beta (float): Weight used for the moving average

    Returns:
        list: List containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)

    return moving_averages
