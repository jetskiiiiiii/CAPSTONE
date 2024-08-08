"""
    Creates a float32 tensor from a Series of 2D matrix to be used in a LSTM model.
"""

import numpy as np


def create_tensor(matrices):
    arrays = []
    for i in matrices:
        arrays.append(i)

    tensor = np.stack(arrays)
    tensor_float32 = tensor.astype(np.float32)

    return tensor_float32
