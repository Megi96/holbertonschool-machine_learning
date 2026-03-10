#!/usr/bin/env python3
"""7-densenet121.py"""
import tensorflow.keras as K


def dense_block(X, layers, growth_rate):
    """
    Builds a dense block as described in DenseNet-121

    X: input tensor
    layers: number of bottleneck layers in the block
    growth_rate: growth rate (number of filters to add per layer)

    Returns: output tensor of the dense block
    """
    for i in range(layers):
        # Bottleneck layer
        bn1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(4 * growth_rate, (1, 1),
                                padding='same',
                                kernel_initializer='he_normal')(act1)

        bn2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(growth_rate, (3, 3),
                                padding='same',
                                kernel_initializer='he_normal')(act2)

        # Concatenate input and new features
        X = K.layers.Concatenate()([X, conv2])

    return X


def transition_layer(X, reduction):
    """
    Builds a transition layer
    X: input tensor
    reduction: compression factor (0<reduction<=1)
    """
    bn = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(bn)
    filters = int(K.backend.int_shape(X)[-1] * reduction)
    conv = K.layers.Conv2D(filters, (1, 1),
                           padding='same',
                           kernel_initializer='he_normal')(act)
    avg = K.layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=2,
                                    padding='same')(conv)
    return avg


def densenet121():
    """
    Builds the DenseNet-121 architecture as described in the original paper
    Returns: Keras model
    """
    growth_rate = 32
    compression = 0.5  # reduction in transition layers
    X_input = K.Input(shape=(224, 224, 3))

    # Initial convolution + pooling
    X = K.layers.Conv2D(64, (7, 7),
                        strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=2,
                              padding='same')(X)

    # Dense Block 1
    X = dense_block(X, layers=6, growth_rate=growth_rate)
    X = transition_layer(X, reduction=compression)

    # Dense Block 2
    X = dense_block(X, layers=12, growth_rate=growth_rate)
    X = transition_layer(X, reduction=compression)

    # Dense Block 3
    X = dense_block(X, layers=24, growth_rate=growth_rate)
    X = transition_layer(X, reduction=compression)

    # Dense Block 4 (no transition)
    X = dense_block(X, layers=16, growth_rate=growth_rate)

    # Final batch norm + ReLU
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    # Global average pooling
    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  padding='same')(X)

    # Dense classifier (1000 classes)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(X)

    model = K.Model(inputs=X_input, outputs=X)
    return model
