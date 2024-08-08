"""
    Adds padding to a single sequence.
"""


def pad_sequence(sequence, max_len):
    while len(sequence) < max_len:
        sequence += "P"
    return sequence
