#!/usr/bin/env python3
"""
5-dense_block.py
Builds a Dense Block as described in Densely Connected Convolutional Networks (DenseNet-B)
"""

import tensorflow as tf
from tensorflow.keras import layers, initializers

def dense_block(X, nb_filters, growth_rate, layers_count):
    """
    Builds a dense block
    
    Parameters
    ----------
    X : tf.Tensor
        Input tensor from previous layer
    nb_filters : int
        Number of filters in input tensor
    growth_rate : int
        Growth rate for the dense block
    layers_count : int
        Number of layers in the dense block
    
    Returns
    -------
    X : tf.Tensor
        Concatenated output of all layers in the dense block
    nb_filters : int
        Number of filters in the concatenated output
    """
    for i in range(layers_count):
        # BatchNorm + ReLU
        bn = layers.BatchNormalization()(X)
        act = layers.Activation('relu')(bn)
        
        # Bottleneck layer: 1x1 Conv
        conv1 = layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=1,
            padding='same',
            kernel_initializer=initializers.he_normal(seed=0)
        )(act)
        
        # BatchNorm + ReLU
        bn2 = layers.BatchNormalization()(conv1)
        act2 = layers.Activation('relu')(bn2)
        
        # 3x3 Conv
        conv2 = layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=initializers.he_normal(seed=0)
        )(act2)
        
        # Concatenate input with new feature maps
        X = layers.Concatenate()([X, conv2])
        
        # Update the number of filters
        nb_filters += growth_rate

    return X, nb_filters
