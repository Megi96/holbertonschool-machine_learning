#Task 12
#!/usr/bin/env python3
"""
12-test.py
Tests a neural network model
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Parameters:
    network: the model to test
    data: input data
    labels: one-hot labels
    verbose: determines verbosity of output

    Returns:
    the loss and accuracy of the model, respectively
    """
    return network.evaluate(
        data,
        labels,
        verbose=verbose
    )
