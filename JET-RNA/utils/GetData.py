import pandas as pd
import tensorflow as tf
from utils.GetDataUtils import *


def get_data(
    data: pd.DataFrame,
    columns: list = ["none"],
    split: bool = False,
    num_samples: int = 0,
    random_state: int = -1,
    max_len: int = 0,
    min_dist: int = 3,
    create_tensor_sequence: bool = False,
    create_tensor_structure: bool = False,
):
    """
    Notes:
    - Order of returns: create_tensor_sequence, create_tensor_structure, data columns
    - get_one_hot_sequence was used for LSTM model to create Lx4 tensors, but now LxLx8 tensors are used

    Args:
        columns: List of column names to be returned (list).
        (name, length, sequence, pairings)

    Returns:
        sequence_tensor:
        structure_tensor:
        columns:

    Raises:
        ValueError: If 'num_samples' is greater than number of total samples in data.
            - 28,370 samples in bpRNA.

    To do:
    - create a for_test option for that will return structure_matrix
        - automatically return sequence_tensor
    - if create_tensor options are not selected, columns can not be empty
    """

    ### Raises
    if (
        "name" not in columns
        and "length" not in columns
        and "sequence" not in columns
        and "pairings" not in columns
        and "none" not in columns
    ):
        raise ValueError(
            "'columns' must be a list with the following options: name, length, sequence, pairings, none (default, for none)"
        )
    if num_samples > len(data):
        raise ValueError("'num_samples' exceeds number of samples in data")

    if split and not columns:
        raise ValueError("'columns' can not be empty if 'split' is True")
    ###

    if max_len != 0:
        data = data[data["length"] <= max_len]

    if num_samples != 0 and random_state > -1:
        data = data.sample(n=num_samples, random_state=random_state)
    # If random_state is -1, get the first x samples
    elif num_samples != 0:
        data = data.head(num_samples)

    num_samples = data.shape[0]

    # If pad is True, max_len must be set
    max_len = data["length"].max() if (max_len == 0) else max_len

    returns = []

    processed_sequences_collection = []
    processed_structures_collection = []

    ### Sequences
    if create_tensor_sequence:
        for idx in range(num_samples):
            dim = data["length"].iloc[idx]
            tensor = build_feature_tensor(data["sequence"].iloc[idx], dim)
            padded_tensor = pad_feature_tensor(tensor, max_len, dim)
            reshaped_tensor = tf.transpose(padded_tensor, perm=[1, 2, 0])

            processed_sequences_collection.append(reshaped_tensor)

        # sequence_tensor = create_tensor(processed_sequences_collection)
        # returns.append(sequence_tensor)
        returns.append(processed_sequences_collection)

    ### Structures
    if create_tensor_structure:
        for idx in range(num_samples):
            pairing = data["pairings"].iloc[idx]
            length = data["length"].iloc[idx]
            pairings = get_pairings_from_structure(pairing)
            matrix = build_matrix(pairings, length)
            matrix_expand = tf.broadcast_to(matrix, (1, length, length))
            padded_matrix = pad_feature_tensor(matrix_expand, max_len, length)
            reshaped_matrix = tf.reshape(padded_matrix, [max_len, max_len, 1])

            processed_structures_collection.append(reshaped_matrix)

        # structure_tensor = create_tensor(processed_structures_collection)
        # returns.append(structure_tensor)
        returns.append(processed_structures_collection)

    if "none" not in columns:
        if split:
            for column in columns:
                returns.append(data[column])
        elif not split:
            returns.append(data[columns])

    return returns[0] if len(returns) == 1 else returns
