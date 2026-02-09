#!/usr/bin/env python3
"""F1 score"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision

import numpy as np


def f1_score(confusion):
    """
    Calculates the F1 score for each class.
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)

    numerator = 2 * prec * sens
    denominator = prec + sens

    # Avoid division by zero (though in this dataset it shouldn't happen)
    f1 = np.where(denominator == 0, 0.0, numerator / denominator)

    return f1
