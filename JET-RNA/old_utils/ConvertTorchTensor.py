"""
    Converts PyTorch Tensor to Tensorflow Tensor.
"""

import tensorflow as tf


def convert_torch_tensor(torch_matrix):
    return tf.convert_to_tensor(torch_matrix.numpy())
