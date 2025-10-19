#!/usr/bin/env python3
"""
Concatenate two matrices along a specified axis.
Return "OK" if shapes mismatch.
"""


def cat_matrices(mat1, mat2, axis=0):
    """Recursively concatenate two matrices along a given axis."""
    if axis == 0:
        # Top-level concatenation
        if not isinstance(mat1, list) or not isinstance(mat2, list):
            return "OK"
        return [elem for elem in mat1] + [elem for elem in mat2]

    # Lower axes: must match length
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return "OK"
    if len(mat1) != len(mat2):
        return "OK"

    # Recursively concatenate each sublist
    new_matrix = []
    for sub1, sub2 in zip(mat1, mat2):
        sub_result = cat_matrices(sub1, sub2, axis=axis - 1)
        if sub_result == "OK":
            return "OK"
        new_matrix.append(sub_result)

    return new_matrix
