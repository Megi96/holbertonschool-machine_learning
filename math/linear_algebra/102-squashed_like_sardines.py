#!/usr/bin/env python3
"""
Concatenate two matrices along a specified axis.
Return "OK" if the shapes do not match for concatenation.
"""


def cat_matrices(mat1, mat2, axis=0):
    """Recursively concatenate mat1 and mat2 along the given axis."""
    # Base case: top-level concatenation (axis 0)
    if axis == 0:
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return "OK"
        return [*mat1, *mat2]

    # Lower axes: dimensions must match
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return "OK"
    if len(mat1) != len(mat2):
        return "OK"

    # Recursively concatenate each corresponding sublist
    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis=axis - 1)
        if sub_result == "OK":
            return "OK"
        new_matrix.append(sub_result)

    return new_matrix
