#!/usr/bin/env python3
"""
Image random cropping utility.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Perform a random crop on an image tensor.

    This function randomly crops a 3D TensorFlow image tensor to the
    specified size. The crop is taken from a random location within
    the image boundaries.

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels).
        Represents a single image. Typical dtype is uint8 or float32.

    size : tuple
        A tuple of integers (new_height, new_width, channels)
        specifying the size of the cropped output tensor.
        The number of channels must match the input image.

    Returns
    -------
    tf.Tensor
        A 3D tensor of shape defined by `size`,
        containing the randomly cropped image.

    Raises
    ------
    ValueError
        If `size` is larger than the input image dimensions.

    Notes
    -----
    - This function uses `tf.image.random_crop`.
    - Commonly used for data augmentation in CNN training.
    - The original image tensor is not modified.

    Example
    -------
    >>> cropped = crop_image(image, (224, 224, 3))
    >>> cropped.shape
    TensorShape([224, 224, 3])
    """
    return tf.image.random_crop(image, size=size)
