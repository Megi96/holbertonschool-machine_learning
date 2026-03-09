#!/usr/bin/env python3
"""
6-transition_layer.py
Builds a transition layer for DenseNet-C
"""

import tensorflow.keras as K

def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer for DenseNet with compression.

    Parameters
    ----------
    X : tensor
        Input tensor from previous layer
    nb_filters : int
        Number of filters in input X
    compression : float
        Compression factor for the number of filters

    Returns
    -------
    X : tensor
        Output of the transition layer
    nb_filters : int
        Number of filters in the output
    """
    # Compute number of filters after compression
    nb_filters = int(nb_filters * compression)
    
    # Batch normalization + ReLU
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    
    # 1x1 Convolution with he_normal initializer
    X = K.layers.Conv2D(
        nb_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    
    # Average pooling to reduce spatial dimensions by 2
    X = K.layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(X)
    
    return X, nb_filters
