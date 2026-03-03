#!/usr/bin/env python3
"""
Image brightness adjustment utility.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly adjust the brightness of an image.

    This function applies a random brightness transformation to a
    3D TensorFlow image tensor. The brightness delta is sampled
    uniformly from the interval [-max_delta, max_delta].

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels)
        representing a single image. Typical dtype is uint8
        or float32.

    max_delta : float
        Maximum brightness change. Must be non-negative.
        The image brightness will be adjusted by a random
        value in the range [-max_delta, max_delta].

    Returns
    -------
    tf.Tensor
        A 3D tensor of the same shape and dtype as the input
        image, with randomly adjusted brightness.

    Raises
    ------
    ValueError
        If max_delta is negative.

    Notes
    -----
    - Uses `tf.image.random_brightness`.
    - Commonly used in data augmentation pipelines for CNNs.
    - The original image tensor is not modified.

    Example
    -------
    >>> adjusted = change_brightness(image, 0.2)
    >>> adjusted.shape
    TensorShape([height, width, channels])
    """
    return tf.image.random_brightness(image, max_delta=max_delta)
