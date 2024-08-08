import pandas as pd
from GetDataFrombpRNA.BuildMatrix import build_matrix
from GetDataFrombpRNA.GetPairings import get_pairings


def encode_structures(pairings):
    encoded_structures_list = []
    for pairing in pairings:
        couple = get_pairings(pairings)
        length = len(pairing)
        encoded_structures_list.append(build_matrix(couple, length))
    return encoded_structures_list
