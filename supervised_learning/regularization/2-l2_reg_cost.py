#!/usr/bin/env python3
"""L2 Regularization Cost with Keras"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates total cost including L2 regularization"""
    return cost + tf.add_n(model.losses)
  
