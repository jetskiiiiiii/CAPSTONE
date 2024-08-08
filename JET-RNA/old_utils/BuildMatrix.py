"""Builds adjacency matrix from list of pairings.
Code by: INSERT REFERENCE

Returns:
    _type_: _description_
"""

import numpy as np


def build_matrix(pairings, size):
    mat = np.zeros((size, size))

    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    for i, j in pairings:
        mat[i, j] = 1
        mat[j, i] = 1

    return mat
