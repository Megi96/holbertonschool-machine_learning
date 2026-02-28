#!/usr/bin/env python3
"""LeNet-5 architecture using Keras"""

from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified LeNet-5 architecture

    X: K.Input of shape (m, 28, 28, 1)

    Returns:
    Compiled Keras model
    """

    initializer = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # Layer 2: Max Pool
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Conv
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # Layer 4: Max Pool
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Dense
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # Layer 6: Dense
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # Output Layer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
