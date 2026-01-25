#!/usr/bin/env python3
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras
    """
    model = K.Sequential()

    for i in range(len(layers)):
        if i == 0:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_dim=nx
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    units=layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        # Dropout only BETWEEN layers, not after the last one
        if keep_prob < 1 and i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
