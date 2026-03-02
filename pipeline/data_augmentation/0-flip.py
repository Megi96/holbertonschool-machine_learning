#!/usr/bin/env python3
#!/usr/bin/env python3
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: 3D tf.Tensor (height, width, channels)

    Returns:
        Flipped 3D tf.Tensor
    """
    return tf.image.flip_left_right(image)
