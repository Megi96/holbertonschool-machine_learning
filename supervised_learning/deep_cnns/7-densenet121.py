#!/usr/bin/env python3
"""
DenseNet-121 architecture
"""
from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer

def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture.

    Args:
        growth_rate: growth rate for dense blocks
        compression: compression factor for transition layers

    Returns:
        Keras model of DenseNet-121
    """
    input = K.Input(shape=(224, 224, 3))

    # Initial BN → ReLU
    x = K.layers.BatchNormalization()(input)
    x = K.layers.ReLU()(x)

    # Initial Conv + MaxPool
    x = K.layers.Conv2D(
        64, (7, 7), strides=2, padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)
    x = K.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    nb_filters = 64

    # Dense Block 1
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 2
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 3
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 4
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    # Final BN → ReLU → Global Average Pooling
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)
    x = K.layers.GlobalAveragePooling2D()(x)

    # Softmax output
    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=K.initializers.he_normal(seed=0))(x)

    model = K.models.Model(inputs=input, outputs=output)
    return model
