#!/usr/bin/env python3
"""Specificity"""

import numpy as np


def specificity(confusion):
    """Calculate specificity for each class."""
    total = np.sum(confusion)
    fn = np.sum(confusion, axis=1)     # false negatives per class
    fp = np.sum(confusion, axis=0)     # false positives per class
    tn = total - fn - fp + np.diag(confusion)

    return tn / (total - fn)
