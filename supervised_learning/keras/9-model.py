#!/usr/bin/env python3
"""
9-model.py
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    network: the model to save
    filename: path where the model should be saved
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model.

    filename: path to the saved model
    Returns: the loaded model
    """
    return K.models.load_model(filename)
