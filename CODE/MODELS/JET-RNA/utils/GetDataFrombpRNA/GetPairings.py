"""Builds pairings from DBN.
Code by: INSERT REFERENCE

Returns:
    pairings: list of paired bases
"""

import numpy as np


def get_pairings(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the pairings list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [
        index
        for index, bracket in enumerate(structure)
        if bracket == "(" or bracket == "["
    ]
    closed = [
        index
        for index, bracket in enumerate(structure)
        if bracket == ")" or bracket == "]"
    ]

    assert len(opened) == len(closed)

    assigned = []
    pairings = []

    for close_index in closed:
        for open_index in opened:
            if open_index < close_index:
                if open_index not in assigned:
                    candidate = open_index
            else:
                break
        assigned.append(candidate)
        pairings.append([candidate, close_index])

    assert len(pairings) == len(opened)

    return pairings
