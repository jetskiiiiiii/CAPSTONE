"""
    Encodes one-hot-sequences for a Series.
"""

import pandas as pd
from GetDataFrombpRNA.OneHotEncoding import get_one_hot_sequence


def encode_sequences(sequences):
    encoded_sequences_list = []
    for sequence in sequences:
        encoded_sequences_list.append(get_one_hot_sequence(sequence))
    return encoded_sequences_list
