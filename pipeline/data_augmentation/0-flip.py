#!/usr/bin/env python3
import tensorflow as tf


def flip_image(image):
    """
    Flip an image horizontally (left-right).

    This function takes a 3D TensorFlow tensor representing an image
    and returns a new tensor where the image is mirrored along the
    vertical axis (i.e., left and right sides are swapped).

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels).
        The tensor should represent a single image.
        The dtype is typically uint8 or float32.

    Returns
    -------
    tf.Tensor
        A 3D tensor of the same shape and dtype as the input image,
        containing the horizontally flipped image.

    Notes
    -----
    - This operation does not modify the original tensor.
    - Commonly used in data augmentation pipelines for
      training convolutional neural networks (CNNs).
    - Internally uses `tf.image.flip_left_right`.

    Example
    -------
    >>> flipped = flip_image(image)
    >>> flipped.shape == image.shape
    True
    """
    return tf.image.flip_left_right(image)
