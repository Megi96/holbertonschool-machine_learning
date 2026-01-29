#!/usr/bin/env python3
"""
13-predict.py
Makes predictions using a neural network
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Parameters:
        network: the model to make predictions with
        data: input data
        verbose: determines verbosity of output

    Returns:
        the prediction for the data
    """
    return network.predict(
        data,
        verbose=verbose
    )
