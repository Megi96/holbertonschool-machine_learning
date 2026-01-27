#!/usr/bin/env python3
"""
11-config.py
Contains functions to save and load a Keras model's configuration.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.

    Parameters:
    network (K.Model): the model whose configuration should be saved
    filename (str): path to the file where the configuration will be saved

    Returns:
    None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model from a JSON configuration file.

    Parameters:
    filename (str): path to the file containing the model configuration

    Returns:
    K.Model: the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()

    return K.models.model_from_json(config)
