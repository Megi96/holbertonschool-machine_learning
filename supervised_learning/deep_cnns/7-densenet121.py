#!/usr/bin/env python3
"""
7-densenet121.py
Builds the DenseNet-121 architecture using only allowed imports
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture

    Parameters
    ----------
    growth_rate : int
        Growth rate for dense blocks
    compression : float
        Compression factor for transition layers

    Returns
    -------
    model : K.models.Model
        Keras model of DenseNet-121
    """
    # Input
    X_input = K.Input(shape=(224, 224, 3))
    
    # Initial convolution + max pooling
    X = K.layers.BatchNormalization()(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    X = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    
    nb_filters = 64  # after initial conv
    
    # Dense Block 1
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=6)
    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)
    
    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=12)
    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)
    
    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=24)
    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)
    
    # Dense Block 4 (no transition after)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=16)
    
    # Final batch norm, activation, global average pooling, softmax
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(X)
    
    # Create model
    model = K.models.Model(inputs=X_input, outputs=X)
    return model
