#!/usr/bin/env python3
"""
Image horizontal flipping utility.
"""

import tensorflow as tf


def flip_image(image):
    """
    Horizontally flip a 3D image tensor.

    This function mirrors an image along its vertical axis,
    effectively swapping the left and right sides of the image.
    The transformation is deterministic and does not alter
    the original tensor.

    Parameters
    ----------
    image : tf.Tensor
        A 3-dimensional tensor of shape (height, width, channels)
        representing a single image. Expected dtypes include
        uint8 or float32.

    Returns
    -------
    tf.Tensor
        A tensor of identical shape and dtype as `image`,
        containing the horizontally flipped result.

    Notes
    -----
    - Uses `tf.image.flip_left_right`.
    - Does not modify the input tensor in-place.
    - Commonly used in data augmentation pipelines for
      convolutional neural networks.
    - Works for any number of channels (e.g., RGB or grayscale).

    Example
    -------
    >>> flipped = flip_image(image)
    >>> flipped.shape
    TensorShape([height, width, channels])
    """
    return tf.image.flip_left_right(image)
