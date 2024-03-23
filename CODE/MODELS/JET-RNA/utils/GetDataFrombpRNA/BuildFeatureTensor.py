"""
    Creates a LxLx8 feature tensor for a sequence.

    Args: sequence (str), onehot (bool), min_dist (int)

    Returns: PyTorch Tensor (which is eventually converted to Tensorflow Tensor)
"""

import torch
import torch.nn as nn
from torch import cuda

# CNNFold
bases = ["A", "U", "G", "C"]
device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")


def build_feature_tensor(sequence, onehot, min_dist, add_padding):
    """
    Create input matrix which is 16xNxN or 1xNxN according to the onehot value
    At the moment works faster than the previous one (matrix multiplication vs normal loop)

    We have 16 different pairing types w.r.t [A, U, G, C]
        0, 5, 10, 15 are self_loops (unpair) --> 1
        1, 4, 6, 9, 11, 14 are pairings --> 6
        others are invalid --> 1
        = 8 modes (channels)

    matrix[0] = itself
    matrix[1-6] = pairings
    matrix[7] = invalid
    """
    n = len(sequence)
    invalid = []
    seq = []
    for i, s in enumerate(sequence):
        if s not in bases:
            invalid.append(i)
            seq.append(0)
        else:
            seq.append(bases.index(s))

    seq = torch.tensor(seq, device=device)
    if onehot:
        mat = torch.zeros((17, n, n), device=device)
    else:
        mat = torch.zeros((1, n, n), device=device)

    q2 = seq.repeat(n, 1)
    q1 = q2.transpose(1, 0)
    t = torch.stack(
        ((torch.abs(q1 - q2) == 1).long(), torch.eye(n, device=device).long())
    )
    mask = torch.max(t, 0)[0]
    flat_mat = (q1 * 4 + q2 + 1) * mask

    for i in range(1, min_dist + 1):
        flat_mat[range(n - i), range(i, n)] = 0
        flat_mat[range(i, n), range(n - i)] = 0

    flat_mat = flat_mat.unsqueeze(0)

    if onehot:
        idx2 = torch.arange(n).repeat(n, 1)
        idx1 = idx2.transpose(1, 0).reshape(-1)
        idx2 = idx2.reshape(-1)
        mat[flat_mat.reshape(-1), idx1, idx2] = 1
        mat = mat[1:]
        mat8 = mat[[1, 4, 6, 9, 11, 14]]

        mat8 = torch.cat((mat8, torch.sum(mat[[0, 5, 10, 15]], 0).unsqueeze(0)), 0)
        mat8 = torch.cat((mat8, 1 - torch.sum(mat8, 0).unsqueeze(0)), 0)
        return mat8

    return flat_mat
