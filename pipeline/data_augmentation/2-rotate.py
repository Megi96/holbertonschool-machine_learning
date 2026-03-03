#!/usr/bin/env python3
"""
Image rotation utility.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotate an image by 90 degrees counterclockwise.

    This function takes a 3D TensorFlow tensor representing a single
    image and rotates it 90 degrees in the counterclockwise direction.
    The rotation is performed using TensorFlow's `tf.image.rot90`.

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels).
        Represents a single image. Common dtypes are uint8 or float32.

    Returns
    -------
    tf.Tensor
        A 3D tensor of the same dtype as the input, rotated
        90 degrees counterclockwise. The resulting shape will be
        (width, height, channels).

    Notes
    -----
    - This operation does not modify the original tensor.
    - Uses `tf.image.rot90` with the default parameter k=1.
    - Frequently used in data augmentation pipelines for
      convolutional neural networks (CNNs).

    Example
    -------
    >>> rotated = rotate_image(image)
    >>> rotated.shape
    TensorShape([width, height, channels])
    """
    return tf.image.rot90(image)
