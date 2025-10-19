#!/usr/bin/env python3
"""
Concatenate two nested lists (matrices) along a specified axis.
Return "OK" if shapes are incompatible.
"""


def cat_matrices(mat1, mat2, axis=0):
    """Recursively concatenate mat1 and mat2 along the given axis."""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        # Reached innermost elements
        return None

    if axis == 0:
        # Top-level concatenation
        if len(mat1) != len(mat1) and len(mat2) != len(mat2):
            return "OK"
        return [*mat1, *mat2]

    # Check lower dimensions: must have same length
    if len(mat1) != len(mat2):
        return "OK"

    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis - 1)
        if sub_result == "OK":
            return "OK"
        new_matrix.append(sub_result)

    return new_matrix
