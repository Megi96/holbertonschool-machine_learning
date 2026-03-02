#!/usr/bin/env python3
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Args:
        image: 3D tf.Tensor (height, width, channels)

    Returns:
        Rotated 3D tf.Tensor
    """
    return tf.image.rot90(image)
