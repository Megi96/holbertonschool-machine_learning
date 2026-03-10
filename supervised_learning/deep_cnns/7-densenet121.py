#!/usr/bin/env python3
"""DenseNet-121 Architecture Implementation."""

import tensorflow.keras as K


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Args:
        growth_rate (int): The growth rate for dense blocks.
        compression (float): The compression factor for transition layers.

    Returns:
        keras.Model: The DenseNet-121 model.
    """
    dense_block = __import__('5-dense_block').dense_block
    transition_layer = __import__('6-transition_layer').transition_layer

    # He normal initializer with seed=0
    he_init = K.initializers.HeNormal(seed=0)

    # Input layer
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution (7x7, stride 2, 64 filters)
    # BatchNorm -> ReLU -> Conv
    X = K.layers.BatchNormalization()(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=he_init
    )(X)

    # Max pooling (3x3, stride 2)
    X = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same'
    )(X)

    # DenseNet-121 configuration: [6, 12, 24, 16]
    num_layers = [6, 12, 24, 16]

    # Build dense blocks with transition layers between them
    for i, num_layers_block in enumerate(num_layers):
        # Dense block
        X, num_filters = dense_block(X, X.shape[-1], growth_rate, num_layers_block)

        # Transition layer after each dense block except the last one
        if i < len(num_layers) - 1:
            X, num_filters = transition_layer(X, num_filters, compression)

    # Final BatchNorm and ReLU
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    # Average pooling (7x7 to get to 1x1)
    X = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='valid'
    )(X)

    # Flatten
    X = K.layers.Flatten()(X)

    # Fully connected layer with 1000 units (ImageNet classes)
    outputs = K.layers.Dense(
        units=1000,
        kernel_initializer=he_init
    )(X)

    # Create model
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
