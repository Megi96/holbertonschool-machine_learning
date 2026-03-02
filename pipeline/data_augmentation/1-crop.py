#!/usr/bin/env python3
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: 3D tf.Tensor (height, width, channels)
        size: tuple (new_height, new_width, channels)

    Returns:
        Cropped 3D tf.Tensor
    """
    return tf.image.random_crop(image, size=size)
