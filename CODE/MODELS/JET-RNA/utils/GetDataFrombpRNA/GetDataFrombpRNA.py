"""
    - 'columns' options: 'name', 'length', 'sequence', 'pairings', 'structure', 'pseudoknot', 'sequence_padded',
    'sequence_one_hot', 'structure_padded', 'structure_matrix'
    - 'num_samples' must be less than or equal to 28,370 (# of samples in bpRNA)
    - 'max_len' filters out sequences that are too long
        - if max_len = 0 but add_padding = True, max_len should be set to longest length
    - if 'random_state' is negative, the first x samples are taken
        - if positive, random sampling is implemented and 'random_state' ensures results can be reproduced
    - 'encode_one_hot', 'encode_pairing_matrix', 'add_padding' creates separate columns in bpRNA DataFrame
        - before being encoded, check first if sequences/structures need padding
    - If create_tensor_sequence or create_tensor_structure is True, the sequences/structures must be encoded
        - however, the encoded sequence/structure doens't need to be returned
    - For now, currently only accepts lstm and cnn model data processing
    
    Returns:
    - any columns options
    - sequence_tensor, structure_tensor
"""

import pandas as pd
from GetDataFrombpRNA.BuildMatrixSeries import build_matrix_series
from GetDataFrombpRNA.CreateTensor import create_tensor
from GetDataFrombpRNA.EncodeSequences import encode_sequences
from GetDataFrombpRNA.PadSequenceSeries import pad_sequence_series
from GetDataFrombpRNA.PadStructureSeries import pad_structure_series
from GetDataFrombpRNA.BuildFeatureTensorSeries import build_feature_tensor_series


def get_data_from_bprna(
    bprna,
    model,
    columns="none",
    num_samples=0,
    max_len=0,
    min_dist=3,
    random_state=-1,
    encode_one_hot=False,
    encode_pairing_matrix=False,
    add_padding=False,
    create_tensor_sequence=False,
    create_tensor_structure=False,
    split=False,
):
    if isinstance(bprna, str):
        bprna = pd.read_csv(f"{bprna}")
    elif isinstance(bprna, pd.DataFrame):
        bprna = bprna
    else:
        raise TypeError("'bprna' must be pd.DataFrame or path to csv")
    if model != "cnn" and model != "lstm":
        raise ValueError("'model' must either be 'cnn' or 'lstm'")
    if (
        model == "cnn"
        and not add_padding
        and create_tensor_structure
        and num_samples > 1
    ):
        raise ValueError("add_padding must be true for 'cnn' model tensors")
    if columns != "none" and columns != "all" and not isinstance(columns, list):
        raise ValueError("'columns' must be a list or 'all' or 'none'")
    if not isinstance(num_samples, int):
        raise TypeError("'num_samples' must be an integer")
    if num_samples > 28370:
        raise ValueError("'num_samples' exceeds number of samples in bpRNA")
    if not isinstance(max_len, int):
        raise TypeError("'max_len' must be an integer")
    if not isinstance(random_state, int):
        raise TypeError("'random_state' must be an integer")
    if not isinstance(add_padding, bool):
        raise TypeError("'add_padding' must be a boolean")
    if not isinstance(create_tensor_sequence, bool):
        raise TypeError("'create_tensor' must be a boolean")
    if not isinstance(create_tensor_structure, bool):
        raise TypeError("'create_tensor' must be a boolean")
    if create_tensor_sequence:
        if not encode_one_hot and model == "lstm":
            raise ValueError("'encode_one_hot' must be True to create sequence tensor")
        elif not add_padding and num_samples and model == "lstm":
            raise ValueError("'add_padding' must be True if tensor")
    if create_tensor_structure:
        if not encode_pairing_matrix:
            raise ValueError(
                "'encode_pairing_matrix' must be True to create structure tensor"
            )
        elif not add_padding:
            raise ValueError("'add_padding' must be True to create tensor")

    if max_len != 0:
        bprna = bprna[bprna["length"] <= max_len]

    if num_samples != 0:
        if random_state < 0:
            bprna = bprna.head(num_samples)
        else:
            bprna = bprna.sample(n=num_samples, random_state=random_state)

    # for if add_padding is True but max_len isn't set
    max_len = bprna["length"].max() if (max_len == 0) else max_len

    if add_padding:
        sequence_padded = pad_sequence_series(bprna["sequence"], max_len)
        bprna["sequence_padded"] = sequence_padded

        structure_padded = pad_structure_series(bprna["structure"], max_len)
        bprna["structure_padded"] = structure_padded

    if encode_one_hot:
        sequences = bprna["sequence_padded"] if add_padding else bprna["sequence"]
        sequence_one_hot = encode_sequences(sequences)
        bprna["sequence_one_hot"] = sequence_one_hot

    if encode_pairing_matrix:
        lengths = max_len if add_padding else bprna["length"]
        structures = bprna["structure_padded"] if add_padding else bprna["structure"]
        structure_matrix = build_matrix_series(structures, lengths, model)
        bprna["structure_matrix"] = structure_matrix

    returns = []

    # 'columns' can be split
    if isinstance(columns, list) and split:
        for item in columns:
            returns.append(bprna[item])
    elif isinstance(columns, list) and not split:
        returns.append(bprna[columns])
    elif columns == "all":
        returns.append(bprna)
    elif columns == "none":
        pass

    # sequence always goes first in the return, then structure
    if model == "lstm" and create_tensor_sequence:
        sequence_tensor = create_tensor(bprna["sequence_one_hot"])
        returns.append(sequence_tensor)

    ### CNN feature matrix
    if model == "cnn" and create_tensor_sequence:
        # if encode_one_hot is True, feature matrix will be one_hot
        # if add_padding is True, padding will be added later (instead of pre-padding
        # technique on 'lstm' sequence tensors)
        feature_tensors = build_feature_tensor_series(
            bprna["sequence"], encode_one_hot, min_dist, add_padding, max_len
        )
        bprna["feature_tensor"] = feature_tensors
        combined_feature_tensor = create_tensor(bprna["feature_tensor"])
        returns.append(combined_feature_tensor)

    if model == "lstm" and create_tensor_structure:
        structure_tensor = create_tensor(bprna["structure_matrix"])
        returns.append(structure_tensor)

    if model == "cnn" and create_tensor_structure:
        structure_tensor = create_tensor(bprna["structure_matrix"])
        returns.append(structure_tensor)

    return returns[0] if len(returns) == 1 else returns


# import pandas as pd
# import re
# from GetDataFrombpRNA.OneHotEncoding import get_one_hot_sequence
# from GetDataFrombpRNA.BuildMatrixFromDotBracket import get_couples, build_matrix

# """
# Goes through a DBN file and extracts the name, length, sequence, and dot-bracket notation.
# B - bulge
# H - hairpin loop
# M - multiloop
# I - internal loop
# E - external loop (one stem)
# S = stem
# PK - pseudoknot
# NCBP6 - non-Watson Creek base pair
# X - external loop (two stems)
# segment
# """


# def get_name_from_bprna(filename):
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             values = line.strip().split(" ")
#             if values[0].replace("#", "").replace(":", "") == "Name":
#                 return values[-1]


# def get_length_from_bprna(filename):
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             values = line.strip().split(" ")
#             if values[0].replace("#", "").replace(":", "") == "Length":
#                 length = re.sub(r"\D", "", values[-1])
#                 return int(length)


# def get_sequence_from_bprna(filename):
#     bases = {"A", "U", "C", "G", "N"}
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             if line[0].upper() in bases:
#                 return line[:-1]


# def get_pairings_from_bprna(filename):
#     pairings = {".", "(", "["}
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             if line[0] in pairings:
#                 return line[:-1]


# def get_structure_from_bprna(filename):
#     structures = {"X", "E", "H", "S", "I", "M", "B"}
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             if line[0].upper() in structures:
#                 return line[:-1]


# def get_pseudoknot_from_bprna(filename):
#     structures = {"N", "K"}
#     with open(filename, "r") as file:
#         for i, line in enumerate(file):
#             if line[0].upper() in structures:
#                 return line[:-1]


# def get_data_from_bprna(filename):
#     data = {}
#     data["name"] = get_name_from_bprna(filename)
#     data["length"] = get_length_from_bprna(filename)
#     data["sequence"] = get_sequence_from_bprna(filename)
#     # data["one_hot_sequence"] = get_one_hot_sequence(data["sequence"])
#     data["pairings"] = get_pairings_from_bprna(filename)
#     # data["pairing_matrix"] = build_matrix(get_couples(data["pairings"]), data["length"])
#     data["structure"] = get_structure_from_bprna(filename)
#     data["pseudoknot"] = get_pseudoknot_from_bprna(filename)
#     data_df = pd.DataFrame([data])
#     return data_df


# if __name__ == "__main__":
#     pass
