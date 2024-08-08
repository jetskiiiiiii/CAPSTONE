"""
    Adds padding to a Series of sequences.
"""

from GetDataFrombpRNA.PadSequence import pad_sequence


def pad_sequence_series(sequences, max_len):
    padded_sequences = []
    for sequence in sequences:
        padded_sequences.append(pad_sequence(sequence, max_len))
    return padded_sequences
