#!/usr/bin/env python3
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras
    """

    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            # First layer needs input dimension
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_dim=nx
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        # Add dropout after each layer
        if keep_prob < 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
