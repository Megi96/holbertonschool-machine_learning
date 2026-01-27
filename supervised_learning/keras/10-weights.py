#!/usr/bin/env python3
"""
10-weights.py
Functions to save and load Keras model weights.
"""

def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a Keras model.

    Parameters:
    - network: the Keras model whose weights should be saved
    - filename: the path to save the weights
    - save_format: 'keras' (HDF5) or 'tf' (TensorFlow SavedModel)
    
    Returns: None
    """
    # Validate save_format
    if save_format not in ['keras', 'tf']:
        raise ValueError("save_format must be 'keras' or 'tf'")
    
    # Save weights
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads weights into a Keras model.

    Parameters:
    - network: the Keras model to load weights into
    - filename: the path from which to load the weights
    
    Returns: None
    """
    network.load_weights(filename)
