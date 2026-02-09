#!/usr/bin/env python3
"""F1 score"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

import numpy as np


def f1_score(confusion):
    """Calculate F1 score for each class."""
    p = precision(confusion)
    s = sensitivity(confusion)
    return 2 * p * s / (p + s)
