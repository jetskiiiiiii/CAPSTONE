"""After pair prediction, organize the pairiings into secondary/tertiary structural elements
Types of secondary/tertiary stru;ctural elements (definitions from bpRNA):
1. Stem - region of uninterrupted base pairs
2. Hairpin loop - unpaired sequence with both ends meeting at the 2 strands of a stem region (the same stem)
3. Internal loop - 2 unpaired strands closed by base pairs on both sides
4. Bulge - a special case of an internal loop where 1 of the strands is of lenth 0
5. Multiloops - cycle of more than 2 unpaired strands connected by stems
6. External loops - similar to multiloops, but are not connected in a cycle
7. Dangling ends - unpaired strands at the beginning and end of the RNA sequence
Tertiary:
8. Pseudoknots - base pairs that satisfy the PK ordering (i<i'<j<j' or i'<i<j'<j); denoted in 2ndary structure as the minimal set that results in a PK-free structure
9. Base triplet
10. Lone pair -
11. Noncanonical -
"""

import numpy as np
import pandas as pd
from OneHotEncoding import get_one_hot_sequence
from BuildMatrixFromDotBracket import get_couples, build_matrix

structure = ".....(((((((((((((((((((((((....)))))))))).)))))))))))))..(((...))).(((((((....)))))))....................."
couples = get_couples(structure)
mat = build_matrix(couples, len(structure))

mat_flat = mat.flatten()

mat_nonzero = np.any(mat, axis=1)

num_of_columns = mat.shape[0]
count_of_pairs = (
    np.count_nonzero(mat) - ((num_of_columns - 2) * 2) - 2
)  # every column has 2 extra 1s, except 2 columns which only have 1 extra 1s

# export adjacency matrix to csv
# mat_df = pd.DataFrame(mat)
# mat_df.to_csv("./CODE/PRACTICE/RNA/matrix_file.csv", header=False, index=False)

# NOTE: currently, inputs are in the form of dot notation, but goal is to get structure from adjacency matris
# given that the predicted pairs are in the form of adjacency matrix


def get_stems_from_dot_bracket(dot_bracket):
    stems = []
    current_count = 0
    for bases in dot_bracket:
        if bases == ")":
            current_count += 1
        else:
            if current_count > 0:
                stems.append(current_count)
                current_count = 0

            if current_count > 0:
                stems.append(current_count)
                current_count = 0

    return stems


# def get_stems_from_adjacency_matrix(matrix):
#     stems = []
#     current_count = 0
#     for i in matrix:
#         for j in i:
#             if


# print(get_stems_from_adjacency_matrix(structure))
