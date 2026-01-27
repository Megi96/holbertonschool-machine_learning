#!/usr/bin/env python3
"""
10-weights.py
Contains functions to save and load a Keras model's weights.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights to a file.

    Parameters:
    network (K.Model): the model whose weights should be saved
    filename (str): path to the file where the weights will be saved
    save_format (str): format in which the weights should be saved

    Returns:
    None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads weights into a model.

    Parameters:
    network (K.Model): the model to which the weights will be loaded
    filename (str): path to the file containing the weights

    Returns:
    None
    """
    network.load_weights(filename)
