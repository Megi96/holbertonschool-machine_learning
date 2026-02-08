#!/usr/bin/env python3
import tensorflow as tf

def create_batch_norm_layer(prev, n, activation):
    tf.random.set_seed(0)

    dense = tf.keras.layers.Dense(
        n,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0)
    )(prev)

    bn = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,                # ← very common in old tasks
        gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros(),
        fused=False                  # ← sometimes helps reproducibility
    )(dense, training=True)

    return activation(bn) if activation else bn
