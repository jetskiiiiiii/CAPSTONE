import torch
import torch.nn as nn
from torch import cuda

# CNNFold
device = torch.device("cuda") if cuda.is_available() else torch.device("cpu")


def pad_feature_tensor(mat, final_size, insert_mode="m", padding_value=-1):
    """
    mat has size (k, n, n)
    Create a matrix with size (k, final_size, final_size)
    put the mat in it according to insert_mode:
        m: put mat at the center
        lt: left top
        r: random
    """
    n = mat.shape[-1]
    if final_size < n:
        print("Final size should be greater or equal than the current size!")
    final_mat = torch.ones((mat.shape[0], final_size, final_size), device=device)
    final_mat = final_mat * padding_value
    if insert_mode == "m":
        i = final_size // 2 - n // 2
        final_mat[:, i : i + n, i : i + n] = mat
    return final_mat
