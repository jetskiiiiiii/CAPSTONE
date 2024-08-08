import tensorflow as tf
import keras.models
import argparse
from utils.GetData import *
from utils.GetDataUtils import *
from utils.PostProcessing import *
from utils.PostProcessingUtils import *

"""
Notes: 
    - Assumes 200 max length
"""


max_len = 200


def prepare_prediction_input(sequences):
    processed_sequences_collection = []
    dimensions_collection = []
    for idx in range(len(sequences)):
        sequence = sequences[idx]
        dim = len(sequence)
        tensor = build_feature_tensor(sequence, dim)
        padded_tensor = pad_feature_tensor(tensor, max_len, dim)
        reshaped_tensor = tf.transpose(padded_tensor, perm=[1, 2, 0])

        dimensions_collection.append(dim)
        processed_sequences_collection.append(reshaped_tensor)

    sequence_tensor = create_tensor(processed_sequences_collection)

    return sequence_tensor, dimensions_collection


def predict():
    parser = argparse.ArgumentParser(description="Prediction using jetRNA.")
    parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        required=True,
        help="String of sequences, separated by a comma.",
    )
    parser.add_argument(
        "-m",
        "--model_filepath",
        type=str,
        required=True,
        help="HDF5 file (.h5) of the model.",
    )
    parser.add_argument(
        "--plot_dbn_figure",
        action="store_true",
        help="Plots predicted structure as a figure.",
    )
    parser.add_argument(
        "--plot_dbn_graph",
        action="store_true",
        help="Plots predicted structure as a graph.",
    )

    args = parser.parse_args()
    sequences = args.sequences.split(",")
    model_filepath = args.model_filepath
    plot_figure = args.plot_dbn_figure
    plot_graph = args.plot_dbn_graph

    model = keras.models.load_model(model_filepath)

    sequences_tensor, lengths = prepare_prediction_input(sequences)

    # Returns a np.ndarray
    predictions = model.predict(sequences_tensor)

    post = post_processing(predictions, lengths)
    errors = [idx + 1 for idx in post[0]]
    pairings_dbn_collection = post[1]

    for idx in range(len(sequences)):
        dbn = pairings_dbn_collection[idx]
        sequence = sequences[idx]
        print(f"{idx+1}: {dbn}")
        if plot_figure:
            plot_dbn_figure(dbn, sequence)
        if plot_graph:
            plot_dbn_graph(dbn, sequence)

    print(f"Failed to predict: {errors}")

    return None


if __name__ == "__main__":
    predict()
