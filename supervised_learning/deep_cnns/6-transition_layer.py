#!/usr/bin/env python3
"""
6-transition_layer.py
Builds a transition layer for DenseNet-C
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers

def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer
    
    Parameters
    ----------
    X : tf.Tensor
        Input tensor from previous layer
    nb_filters : int
        Number of filters in input tensor
    compression : float
        Compression factor (0 < compression <= 1)
    
    Returns
    -------
    X : tf.Tensor
        Output of the transition layer
    nb_filters : int
        Number of filters in the output
    """
    # Compute the number of filters after compression
    nb_filters = int(nb_filters * compression)
    
    # BatchNorm + ReLU
    bn = layers.BatchNormalization()(X)
    act = layers.Activation('relu')(bn)
    
    # 1x1 Conv with compressed number of filters
    conv = layers.Conv2D(
        filters=nb_filters,
        kernel_size=1,
        padding='same',
        kernel_initializer=initializers.he_normal(seed=0)
    )(act)
    
    # Average Pooling with pool size 2x2 and stride 2 (downsampling)
    X = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    
    return X, nb_filters
