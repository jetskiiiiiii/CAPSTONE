"""
    Builds adjacency matricies from list of pairings from DataFrame.
    
    
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from GetDataFrombpRNA.BuildMatrix import build_matrix
from GetDataFrombpRNA.GetPairings import get_pairings


def build_matrix_series(structures, lengths, model):
    structure_matrix = []
    if model == "cnn":
        if isinstance(lengths, (int, np.int64)):
            for structure in structures:
                pairings = get_pairings(structure)
                # if add_padding is True, then lengths will be max_len
                matrix = build_matrix(pairings, lengths)
                matrix_expand = tf.expand_dims(matrix, axis=-1)
                structure_matrix.append(matrix_expand)
        else:
            for index, structure in structures.items():
                pairings = get_pairings(structure)
                matrix = build_matrix(pairings, lengths[index])
                matrix_expand = tf.expand_dims(matrix, axis=-1)
                structure_matrix.append(matrix_expand)

    if model == "lstm":
        if isinstance(lengths, (int, np.int64)):
            for structure in structures:
                pairings = get_pairings(structure)
                structure_matrix.append(
                    build_matrix(pairings, lengths)
                )  # if add_padding is True, then lengths will be max_len
        else:
            for index, structure in structures.items():
                pairings = get_pairings(structure)
                structure_matrix.append(build_matrix(pairings, lengths[index]))

    return structure_matrix
