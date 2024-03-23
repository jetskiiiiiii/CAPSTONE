"""
    Adds padding to structures in DBN format in a Series.
"""

from GetDataFrombpRNA.PadStructure import pad_structure


def pad_structure_series(structures, max_len):
    structures_padded = []
    for structure in structures:
        structures_padded.append(pad_structure(structure, max_len))
    return structures_padded
