#!/usr/bin/env python3
"""
Image hue adjustment utility.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Adjust the hue of an image.

    This function shifts the hue channel of a 3D TensorFlow image
    tensor by a specified amount. Hue adjustment modifies the color
    tone of the image without changing brightness or contrast.

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels)
        representing a single RGB image. Typical dtype is
        uint8 or float32.

    delta : float
        Amount to shift the hue channel. Must be in the range
        [-1.0, 1.0]. Positive values shift colors forward
        around the HSV color wheel; negative values shift
        them backward.

    Returns
    -------
    tf.Tensor
        A 3D tensor of the same shape and dtype as the input
        image, with adjusted hue.

    Raises
    ------
    ValueError
        If delta is not in the valid range.

    Notes
    -----
    - Uses `tf.image.adjust_hue`.
    - The image must be RGB (3 channels).
    - Internally converts RGB → HSV → RGB.
    - Commonly used for data augmentation in CNN training.
    - The original tensor is not modified.

    Example
    -------
    >>> altered = change_hue(image, 0.2)
    >>> altered.shape
    TensorShape([height, width, channels])
    """
    return tf.image.adjust_hue(image, delta)
