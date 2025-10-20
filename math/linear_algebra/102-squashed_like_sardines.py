#!/usr/bin/env python3
"""
Concatenate two nested lists (matrices) along a specified axis.
Return "OK" if shapes are incompatible.
"""


def cat_matrices(mat1, mat2, axis=0):
    """Recursively concatenate mat1 and mat2 along the given axis."""
    # If we're at the innermost elements
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return "OK" if axis > 0 else [mat1, mat2]

    if axis == 0:
        return mat1 + mat2

    # For lower axes, lengths must match
    if len(mat1) != len(mat2):
        return "OK"

    result = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis - 1)
        if sub_result == "OK":
            return "OK"
        result.append(sub_result)

    return result
