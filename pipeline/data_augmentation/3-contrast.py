#!/usr/bin/env python3
"""
Image contrast adjustment utility.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjust the contrast of an image.

    This function applies a random contrast transformation to a
    3D TensorFlow image tensor. The contrast factor is sampled
    uniformly from the range [lower, upper].

    Parameters
    ----------
    image : tf.Tensor
        A 3D tensor of shape (height, width, channels)
        representing a single image. Typical dtype is uint8
        or float32.

    lower : float
        Lower bound for the random contrast factor.

    upper : float
        Upper bound for the random contrast factor.

    Returns
    -------
    tf.Tensor
        A 3D tensor of the same shape and dtype as the input
        image, with randomly adjusted contrast.

    Raises
    ------
    ValueError
        If lower is greater than upper.

    Notes
    -----
    - Uses `tf.image.random_contrast`.
    - Commonly used in data augmentation pipelines
      for training CNN models.
    - The original image tensor is not modified.

    Example
    -------
    >>> adjusted = change_contrast(image, 0.5, 2.0)
    >>> adjusted.shape
    TensorShape([height, width, channels])
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
