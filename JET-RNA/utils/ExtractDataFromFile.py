import pandas as pd
import re
from GetDataFrombpRNA.OneHotEncoding import get_one_hot_sequence
from GetDataFrombpRNA.BuildMatrixFromDotBracket import get_couples, build_matrix

"""
Goes through a DBN file and extracts the name, length, sequence, and dot-bracket notation.
B - bulge
H - hairpin loop
M - multiloop
I - internal loop
E - external loop (one stem)
S = stem
PK - pseudoknot
NCBP6 - non-Watson Creek base pair
X - external loop (two stems)
segment
"""


def get_name_from_bprna(filename):
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            values = line.strip().split(" ")
            if values[0].replace("#", "").replace(":", "") == "Name":
                return values[-1]


def get_length_from_bprna(filename):
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            values = line.strip().split(" ")
            if values[0].replace("#", "").replace(":", "") == "Length":
                length = re.sub(r"\D", "", values[-1])
                return int(length)


def get_sequence_from_bprna(filename):
    bases = {"A", "U", "C", "G", "N"}
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if line[0].upper() in bases:
                return line[:-1]


def get_pairings_from_bprna(filename):
    pairings = {".", "(", "["}
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if line[0] in pairings:
                return line[:-1]


def get_structure_from_bprna(filename):
    structures = {"X", "E", "H", "S", "I", "M", "B"}
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if line[0].upper() in structures:
                return line[:-1]


def get_pseudoknot_from_bprna(filename):
    structures = {"N", "K"}
    with open(filename, "r") as file:
        for i, line in enumerate(file):
            if line[0].upper() in structures:
                return line[:-1]


def get_data_from_bprna(filename):
    data = {}
    data["name"] = get_name_from_bprna(filename)
    data["length"] = get_length_from_bprna(filename)
    data["sequence"] = get_sequence_from_bprna(filename)
    # data["one_hot_sequence"] = get_one_hot_sequence(data["sequence"])
    data["pairings"] = get_pairings_from_bprna(filename)
    # data["pairing_matrix"] = build_matrix(get_couples(data["pairings"]), data["length"])
    data["structure"] = get_structure_from_bprna(filename)
    data["pseudoknot"] = get_pseudoknot_from_bprna(filename)
    data_df = pd.DataFrame([data])
    return data_df


if __name__ == "__main__":
    pass
