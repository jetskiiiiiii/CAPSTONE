import numpy as np
import pandas as pd
import signal
from utils.PostProcessingUtils import *

"""
- unpad_matrix 
- edmunds_matching_algorithm
- get_pairings_from_matrix
- get_pk_indices
- pairings_to_dbn
- kth_diag_indices
- set_diagonals_to_zero
plot_dbn_graph
plot_dbn_figure
save_dbn_figure
"""


def post_processing(
    predictions: np.ndarray,
    true_lengths: pd.core.frame.DataFrame | list,
):
    """
    Processes the predictions made by model into pairings in dot-bracket notation.

    Notes:
        To save the returned values, use
        'pd.DataFrame(pred_unpad).to_csv(filename, header=False, index=False)'

        Returns list order: errors, unpadded_matrices_collection, edmunds_matrices_collection, pairings_list_collection, pk_indices_collection, pairings_dbn_collection

    Args:
        predictions (np.ndarray): List of predicted target matrices.
        true_lengths (pd.core.frame.DataFrame): List of true lengths.
        return_dbn (bool): If dot-bracket notation is to be returned.
        return_edmunds_matrix (bool): If target matrix is to be returned.
        return_pairings (bool):
        return_pk_indices (bool):

    Returns:
        processed_matrix: Target matrix
        dbn: Dot-bracket notation.
        pk_indices

    Raises:
    """

    returns = []
    errors = []
    unpadded_matrices_collection = []
    diagonal_matrices_collection = []
    edmunds_matrices_collection = []
    symmetric_matrices_collection = []
    pairings_list_collection = []
    pk_indices_collection = []
    pairings_dbn_collection = []

    num_samples = predictions.shape[0]

    true_lengths_as_frame = isinstance(true_lengths, pd.core.frame.DataFrame)
    true_lengths_as_series = isinstance(true_lengths, pd.Series)
    true_lengths_list = (
        true_lengths.values.flatten().tolist()
        if true_lengths_as_frame or true_lengths_as_series
        else true_lengths
    )

    for idx in range(num_samples):
        dim = true_lengths_list[idx]
        prediction = predictions[idx][:, :, 0]

        # Unpad matrices
        unpadded_matrix = unpad_matrix(
            prediction,
            dim=dim,
        )
        unpadded_matrices_collection.append(unpadded_matrix)
        print(f"Prediction {idx + 1}: unpadded.")

        # Get a symmetric matrix
        symmetric_matrix = make_matrix_symmetric(unpadded_matrix)
        symmetric_matrices_collection.append(symmetric_matrix)
        print(f"Prediction {idx + 1}: made matrix symmetric.")

        # Set main diagonals to 0
        diagonal_matrix = set_diagonals_to_zero(symmetric_matrix)
        diagonal_matrices_collection.append(diagonal_matrix)
        print(f"Prediction {idx + 1}: diagonals set to 0.")

        # Apply Edmund's matching algorithm
        edmunds_matrix = edmunds_matching_algorithm(diagonal_matrix, dim)
        edmunds_matrices_collection.append(edmunds_matrix)
        print(f"Prediction {idx + 1}: applied Edmund's matching algorithm.")

        # Get pairings
        pairings_list = get_pairings_from_matrix(edmunds_matrix, dim)
        pairings_list_collection.append(pairings_list)
        print(f"Prediction {idx + 1}: extracted pairings.")

        # Convert pairings to dbn
        dbn, status, pk_idx = pairings_to_dbn(pairings_list, dim)
        pairings_dbn_collection.append(dbn)
        print(f"Prediction {idx + 1}: converted pairings to dbn.")

        # Check if pairings list is empty
        if status:
            if true_lengths_as_series:
                errors.append(true_lengths.iloc[idx])
            else:
                errors.append(idx)

            if status == 1:
                print(f"Prediction {idx + 1}: ERROR 1 - no pairings detected.")
            if status == 2:
                print(f"Prediction {idx + 1}: ERROR 2 - all pairings were pseudoknots.")

        print("\n")

    return (pairings_dbn_collection, errors, pk_idx)
