"""
    Adds padding to a structre in DBN format.
"""


def pad_structure(structure, max_len):
    while len(structure) < max_len:
        structure += "P"
    return structure
