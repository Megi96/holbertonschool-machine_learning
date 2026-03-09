#!/usr/bin/env python3
"""
7-densenet121.py
Builds the DenseNet-121 architecture
"""

import tensorflow as tf
from tensorflow.keras import layers, models, initializers

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
    model : tf.keras.Model
        Keras model of DenseNet-121
    """
    # Input
    X_input = layers.Input(shape=(224, 224, 3))
    
    # Initial convolution (7x7) + MaxPool
    X = layers.BatchNormalization()(X_input)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same',
        kernel_initializer=initializers.he_normal(seed=0)
    )(X)
    X = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    
    nb_filters = 64  # number of filters after first conv
    
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
    
    # Dense Block 4 (final, no transition)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, layers=16)
    
    # Global Average Pooling + Softmax output
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.GlobalAveragePooling2D()(X)
    X = layers.Dense(
        1000, activation='softmax',
        kernel_initializer=initializers.he_normal(seed=0)
    )(X)
    
    # Create model
    model = models.Model(inputs=X_input, outputs=X)
    return model
