"""
  Turns sequences into an LxLx8 feature tensor (series)
"""

import numpy as np

from GetDataFrombpRNA.BuildFeatureTensor import build_feature_tensor
from GetDataFrombpRNA.PadFeatureTensor import pad_feature_tensor
from GetDataFrombpRNA.ConvertTorchTensor import convert_torch_tensor


def build_feature_tensor_series(sequences, onehot, min_dist, add_padding, max_len=0):
    if add_padding and max_len == 0:
        raise ValueError("'max_len' must be larger than 0 if add_padding is True")
    feature_tensors = []

    # should reshape every tensor
    for sequence in sequences:
        feature_tensor = build_feature_tensor(sequence, onehot, min_dist, add_padding)
        # Even though add_padding is False, feature_tensor still gets renamed
        feature_tensor_padded = (
            pad_feature_tensor(feature_tensor, max_len)
            if add_padding
            else feature_tensor
        )
        feature_tensor_padded_converted = convert_torch_tensor(feature_tensor_padded)
        feature_tensor_padded_converted_reshaped = np.transpose(
            feature_tensor_padded_converted, (1, 2, 0)
        )  # reshape tensor
        feature_tensors.append(feature_tensor_padded_converted_reshaped)
    return feature_tensors
