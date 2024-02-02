# code from https://www.kaggle.com/code/theoviel/generating-graph-matrices-from-the-structures/notebook

import numpy as np


def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == "("]
    closed = [idx for idx, i in enumerate(structure) if i == ")"]

    assert len(opened) == len(closed)

    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])

    assert len(couples) == len(opened)

    return couples


def build_matrix(couples, size):
    mat = np.zeros((size, size))

    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    for i, j in couples:
        mat[i, j] = 1
        mat[j, i] = 1

    return mat
