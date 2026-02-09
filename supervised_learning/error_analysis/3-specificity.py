#!/usr/bin/env python3
"""Precision"""

import numpy as np


def precision(confusion):
    """Calculate precision for each class."""
    tp = np.diag(confusion)
    pp = np.sum(confusion, axis=0)
    return tp / pp
