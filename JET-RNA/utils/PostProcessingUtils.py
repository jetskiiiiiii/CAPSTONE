# import biotite
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

"""
Required packages:
  networkx
  biotite
"""

max_pseudoknot_order = 0


def unpad_matrix(matrix: np.ndarray, dim: int):
    """
    ADAPTED FROM:
    Saman Booy, M., Ilin, A. & Orponen, P. RNA secondary structure prediction with convolutional neural networks.
    BMC Bioinformatics 23, 58 (2022). https://doi.org/10.1186/s12859-021-04540-7
    --------------------------------------------------

    Turn a padded matrix into an unpadded one.
    Assumes matrix has center padding.

    Args:
        matrix (numpy.ndarray): Matrix to be unpadded.
        dim (int): Dimension of the original matrix.

    Returns:
        unpadded_matrix (numpy.ndarray): The unpadded matrix.

    Raises:
        ValueError: If dim is larger than the length of the padded matrix.
    """
    current_size = matrix.shape[0]
    if dim > current_size:
        raise ValueError(
            "Length of the final matrix is larger than the length of the padded matrix."
        )

    pad_len = current_size // 2 - dim // 2

    unpadded_matrix = matrix[pad_len : pad_len + dim, pad_len : pad_len + dim]

    return unpadded_matrix


def edmunds_matching_algorithm(matrix: np.ndarray, dim: int):
    """
    ADAPTED FROM:
    Saman Booy, M., Ilin, A. & Orponen, P. RNA secondary structure prediction with convolutional neural networks.
    BMC Bioinformatics 23, 58 (2022). https://doi.org/10.1186/s12859-021-04540-7
    --------------------------------------------------

    Args:
      matrix (numpy.ndarray): Probability matrix indicating pairing possibilities.
      dim (int): Dimension of the original matrix.

    Returns:
      edmunds_matrix (numpy.ndarray): Matrix representing pairings.
    """
    mask = np.eye(dim)
    big_a = np.zeros((2 * dim, 2 * dim))
    big_a[:dim, :dim] = matrix
    big_a[dim:, dim:] = matrix
    big_a[:dim, dim:] = matrix * mask
    big_a[dim:, :dim] = matrix * mask
    G = nx.from_numpy_array(big_a)
    pairings = nx.matching.max_weight_matching(G)
    edmunds_matrix = np.zeros_like(matrix)
    for i, j in pairings:
        if i > dim and j > dim:
            continue
        edmunds_matrix[i % dim, j % dim] = 1
        edmunds_matrix[j % dim, i % dim] = 1

    return edmunds_matrix


def get_pairings_from_matrix(matrix: np.ndarray, dim: int):
    """
    Get pairings from a matching matrix.

    Args:
      matrix (numpy.ndarray): Matrix of 1s and 0s indicating pairs (numpy.ndarray).
      dim (int): Dimensions of matching matrix (int).

    Returns:
      pairings_list (list): List of tuples indicating paired indices (list).
    """
    pairings_list = []

    u = np.triu_indices(dim, k=2)
    u_val = matrix[np.triu_indices(dim, k=2)]
    for idx in range(len(u_val)):
        if u_val[idx] == 1:
            pairings_list.append((u[0][idx], u[1][idx]))

    return pairings_list


def pairings_to_dbn(pairings_list: list, dim: int):
    """
    Converts the pairing indices list into an RNA structure in dot-bracket notation.
    Also gets the pk indices.

    Notes:
        If pairings_list is empty, model has failed to make any predictions, and string of '.'
        will be returned.

        Base pairs on the 3' end have priority, as all other base pairs will be compared to
        existing base pairs.

    Args:
        pairings_list (list): List indicating pairings.
        dim (int): The number of bases in the strand.

    Returns:
        pairings_dbn (str): Pairings in dot_bracket_notation.
        status (int): 1 if no pairings were found, 2 if all pairs formed pseudoknots, 0 otherwise.
        pk_idx (list): List of indices for which bases formed pseudoknots in the pairing list.
    """
    status = 0
    dbn = ["."] * dim
    pk_idx = []

    # No pairings exist, send error
    if not pairings_list:
        status = 1
        return "".join(dbn), status, pk_idx
    b1_list = []
    b2_list = []
    for idx in range(len(pairings_list)):
        b1 = pairings_list[idx][0]
        b2 = pairings_list[idx][1]
        if b1 + 3 >= b2:
            continue
        # First pair assumed to not be pseudoknot
        elif not b1_list:
            dbn[b1] = "("
            b1_list.append(b1)
            dbn[b2] = ")"
            b2_list.append(b2)
        # Detect pseudoknot if i < i' < j < j'
        elif (
            any(b1 > b for b in b1_list)
            and any(b1 < b for b in b2_list)
            and any(b2 > b for b in b2_list)
        ):
            pk_idx.append(idx)
            continue
        # Detect pseudoknot if i' < i < j' < j
        elif (
            any(b1 < b for b in b1_list)
            and any(b2 > b for b in b1_list)
            and any(b2 < b for b in b2_list)
        ):
            pk_idx.append(idx)
            continue
        # If no pseudoknot is formed, append pair to dbn
        else:
            dbn[b1] = "("
            b1_list.append(b1)
            dbn[b2] = ")"
            b2_list.append(b2)

    pairings_dbn = "".join(dbn)

    # No pairings formed, all pseudoknots, send error
    if "(" not in pairings_dbn:
        status = 2
        return pairings_dbn, status, pk_idx

    return pairings_dbn, status, pk_idx


def kth_diag_indices(matrix: np.ndarray, k: int):
    """
    SOURCE: https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    --------------------------------------------------

    Gets indices of the k-th diagonal of a 2x2 matrix.

    Args:
      matrix (np.ndarray): Matrix.
      k (int): Offset from the main diagonal (0 = main diagonal)

    Returns:
      row (np.ndarray): Row values of the indices.
      col (np.ndarray): Column values of the indeces.
    """
    if matrix.ndim != 2:
        raise ValueError("Matrix should be 2D.")

    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def set_diagonals_to_zero(matrix: np.ndarray):
    """
    Sets main diagonal and k=1 offset diagonal to zero.
    This ensure no interference with Edmund's matching algorithm.

    Args:
      matrix (np.ndarray): Probability matrix.

    Returns:
      matrix (np.ndarray): Probability matrix with diagonals set to 0.
    """
    offset = [-2, -1, 0, 1, 2]
    for k in offset:
        matrix[kth_diag_indices(matrix, k)] = 0
    return matrix


def make_matrix_symmetric(matrix: np.ndarray):
    """
    Makes matrix symmetric, using average.

    Args:
        matrix (np.ndarray): Probability matrix

    Returns:
        matrix_symmetric (np.ndarray): Symmetric matrix.
    """
    matrix_transpose = matrix.transpose()
    matrix_symmetric = (matrix + matrix_transpose) / 2
    return matrix_symmetric


def plot_dbn_graph(pairings_list: str, sequence: str, dim: int, filename: str = None):
    """
    ADAPTED FROM: Tom David Müller
    LICENSE: BSD 3 clause
    --------------------------------------------------

    Plots the RNA secondary structure as an arc graph.

    Args:
      pairings_list (list): Pairings as a list of tuples.
      sequence (str): Base names.
      dim (int): Length of sequence.
      filname (str): Path to file (.png).

    Returns:
      None

    Raises:
      TypeError: If pairing
    """

    if not isinstance(pairings_list, np.ndarray):
        raise TypeError("pairing should be of type np.ndarray")
    # Code source: Tom David Müller
    # License: BSD 3 clause
    residue_ids = [num for num in range(dim)]

    # Create a matplotlib pyplot
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    # Setup the axis
    ax.set_xlim(0.5, dim + 0.5)
    ax.set_ylim(0, dim / 2 + 0.5)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_yticks([])

    # Remove the frame
    plt.box(False)

    # Plot the residue names in order
    for residue_name, residue_id in zip(sequence, residue_ids):
        ax.text(residue_id, 0, residue_name, ha="center", fontsize=8)

    # Get the indices of pseudknots
    _, __, pk_idx = pairings_to_dbn(pairings_list, dim)
    # Gets the order of pseudoknots (will be all 0 for now)
    pseudoknot_order = [0] * len(pairings_list)

    # Draw the arcs between base pairs
    for (base1, base2), order in zip(pairings_list, pseudoknot_order):
        base1_res_id = residue_ids[base1]
        base2_res_id = residue_ids[base2]
        arc_center = (np.mean((base1_res_id, base2_res_id)), 1.5)
        arc_diameter = abs(base2_res_id - base1_res_id)
        name1 = sequence[base1]
        name2 = sequence[base2]
        if sorted([name1, name2]) in [["A", "U"], ["C", "G"]]:
            color = biotite.colors["dimorange"]
        else:
            color = biotite.colors["brightorange"]
        if order == 0:
            linestyle = "-"
        elif order == 1:
            linestyle = "--"
        else:
            linestyle = ":"
        arc = matplotlib.patches.Arc(
            arc_center,
            arc_diameter,
            arc_diameter,
            theta1=0,
            theta2=180,
            color=color,
            linewidth=1.5,
            linestyle=linestyle,
        )
        ax.add_patch(arc)

    if filename:
        plt.savefig(filename)

    # Display the plot
    plt.show()

    return None


def get_predictions_as_files(
    predictions: np.ndarray,
    true_lengths: pd.core.frame.DataFrame | list,
    i: int = 0,
    filename_prefix: str = "",
):
    """
    Gets prediction in 4 versions: original, unpadded, diagonal, and edmunds.

    Args:
        predictions (np.ndarray): List of predicted target matrices.
        true_lengths (list | pd.Series): List of true lengths.
        i (int): Index of desired sequence.

    Returns:
        None
    """
    if i > predictions.shape[0]:
        raise IndexError(f"Index must be between 0 and {predictions.shape[0]-1}")

    true_lengths_as_frame = isinstance(true_lengths, pd.core.frame.DataFrame)
    true_lengths_as_series = isinstance(true_lengths, pd.Series)
    true_lengths_list = (
        true_lengths.values.flatten().tolist()
        if true_lengths_as_frame or true_lengths_as_series
        else true_lengths
    )

    dim = true_lengths_list[i]
    name = true_lengths.index[i]
    if not filename_prefix:
        filename_prefix = f"pred"

    # Original
    pred = predictions[i][:, :, 0]
    pd.DataFrame(pred).to_csv(
        f"{filename_prefix}_{name}.csv", header=False, index=False
    )

    # Unpad
    pred_unpad = unpad_matrix(pred, dim)
    pd.DataFrame(pred_unpad).to_csv(
        f"{filename_prefix}_{name}_unpad.csv", header=False, index=False
    )

    # Symmetry
    pred_symmetry = make_matrix_symmetric(pred_unpad)
    pd.DataFrame(pred_symmetry).to_csv(
        f"{filename_prefix}_{name}_symmetry.csv", header=False, index=False
    )

    # Set diagonal to zero
    pred_diag = set_diagonals_to_zero(pred_symmetry)
    pd.DataFrame(pred_diag).to_csv(
        f"{filename_prefix}_{name}_diag.csv", header=False, index=False
    )

    # Apply Edmund's matching algorithm
    pred_edmunds_matrix = edmunds_matching_algorithm(pred_diag, dim)
    pd.DataFrame(pred_edmunds_matrix).to_csv(
        f"{filename_prefix}_{name}_edmunds_matrix.csv", header=False, index=False
    )

    return None
