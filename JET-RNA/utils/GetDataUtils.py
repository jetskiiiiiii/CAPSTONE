import tensorflow as tf


def create_tensor(processed_collection: list):
    """
    Creates a float32 tensor from a Series of 2D matrix.

    Args:
        processed_collection (list): List of tensors of individual sequences.

    Returns:
        tensor (tf.data.Dataset): Tensor of all samples.
    """

    tensor = tf.data.Dataset.from_tensors(processed_collection)

    return tensor


def build_feature_tensor(sequence: str, dim: int):
    """
    ADAPTED FROM:
    Saman Booy, M., Ilin, A. & Orponen, P. RNA secondary structure prediction with convolutional neural networks.
    BMC Bioinformatics 23, 58 (2022). https://doi.org/10.1186/s12859-021-04540-7

        "We have 16 different pairing types w.r.t [A, U, G, C]
            0, 5, 10, 15 are self_loops (unpair) --> 1
            1, 4, 6, 9, 11, 14 are pairings --> 6
            others are invalid --> 1
            = 8 modes (channels)

        matrix[0] = itself
        matrix[1-6] = pairings
        matrix[7] = invalid
        Creates a LxLx8 feature tensor for a sequence."
    --------------------------------------------------

    Builds a feature tensor of shape (8, dim, dim) which indicate potential pairing possibilities.
    Assumes minimum distance of 3.

    Notes:
        Because Tensorflow Tensor objects don't support item assignment, tensors must first be converted
        to numpy tensors and then assigned and then converted back to tensors.

    Args:
        sequence (str):
        dim (int):

    Returns:
        flat_mat (tf.Tensor):

    """
    bases = ["A", "U", "G", "C"]
    min_dist = 3
    invalid = []
    seq = []
    for i, s in enumerate(sequence):
        if s not in bases:
            invalid.append(i)
            seq.append(0)
        else:
            seq.append(bases.index(s))

    seq = tf.convert_to_tensor(seq)

    q2 = tf.stack([seq] * dim)
    q1 = tf.transpose(q2, perm=[1, 0])
    x = tf.cast((tf.math.abs(q1 - q2) == 1), dtype=tf.int32).numpy()
    t = tf.stack((x, tf.eye(dim, dtype=tf.int32)))
    mask = tf.math.maximum(t, 0).numpy()[0]
    flat_mat = (q1 * 4 + q2 + 1) * mask

    flat_mat = flat_mat.numpy()
    for i in range(1, min_dist + 1):
        flat_mat[range(dim - i), range(i, dim)] = 0
        flat_mat[range(i, dim), range(dim - i)] = 0
    flat_mat = tf.convert_to_tensor(flat_mat)

    flat_mat = tf.expand_dims(flat_mat, axis=0)
    mat = tf.zeros((17, dim, dim))
    idx2 = tf.tile(tf.reshape(tf.range(dim), [1, -1]), [dim, 1])
    idx1 = tf.reshape(tf.transpose(idx2, perm=[1, 0]), [-1])
    idx2 = tf.reshape(idx2, [-1])

    mat = mat.numpy()
    mat[tf.reshape(flat_mat, [-1]), idx1, idx2] = 1
    mat = tf.convert_to_tensor(mat)

    mat = mat[1:]
    row_sum = tf.math.reduce_sum(tf.gather(mat, [0, 5, 10, 15], axis=0), axis=0)
    expanded_sum = tf.expand_dims(row_sum, axis=0)

    mat = tf.unstack(mat)
    mat8 = tf.stack([mat[1], mat[4], mat[6], mat[9], mat[11], mat[14]])

    mat8_1 = tf.concat([mat8, expanded_sum], axis=0)

    row_sum_2 = 1 - tf.math.reduce_sum(mat8, axis=0)
    expanded_sum_2 = tf.expand_dims(row_sum_2, axis=0)
    mat8_2 = tf.concat([mat8_1, expanded_sum_2], axis=0)
    return mat8_2


def build_matrix(pairings_list: list, dim: int):
    """
    Code by: INSERT REFERENCE
    --------------------------------------------------

    Builds adjacency matrix from list of pairings.

    Args:
        pairings_list (list): List of pairings.
        dim (int): Length of sequence.

    Returns:
        matrix (tf.Tensor): Adjacency matrix indicating pairings.
    """
    matrix = tf.zeros((dim, dim))
    matrix = matrix.numpy()

    for i in range(dim):  # neigbouring bases are linked as well
        if i < dim - 1:

            matrix[i, i + 1] = 1
        if i > 0:
            matrix[i, i - 1] = 1

    for i, j in pairings_list:
        matrix[i, j] = 1
        matrix[j, i] = 1

    matrix = tf.convert_to_tensor(matrix)
    return matrix


def get_pairings_from_structure(structure: str):
    """
    SOURCE:
        "For each closing parenthesis, I find the matching opening one and store their index in the pairings list.
        The assigned list is used to keep track of the assigned opening parenthesis"
    --------------------------------------------------

    Args:
        structure (str): Pairings in dot-bracket notation.

    Returns:
        pairings (list): List of tuples containing indices of paired bases.
    """
    opened = [
        index
        for index, bracket in enumerate(structure)
        if bracket == "(" or bracket == "["
    ]
    closed = [
        index
        for index, bracket in enumerate(structure)
        if bracket == ")" or bracket == "]"
    ]

    assert len(opened) == len(closed)

    assigned = []
    pairings = []

    for close_index in closed:
        for open_index in opened:
            if open_index < close_index:
                if open_index not in assigned:
                    candidate = open_index
            else:
                break
        assigned.append(candidate)
        pairings.append([candidate, close_index])

    assert len(pairings) == len(opened)

    return pairings


def pad_feature_tensor(tensor: tf.Tensor, final_size: int, dim: int):
    """
    ADAPTED FROM:
    Saman Booy, M., Ilin, A. & Orponen, P. RNA secondary structure prediction with convolutional neural networks.
    BMC Bioinformatics 23, 58 (2022). https://doi.org/10.1186/s12859-021-04540-7
    --------------------------------------------------

    Takes a tensor of shape (k, n, n) and pads it into a tensor of shape (k, final_size, final_size).
    Places the tensor in the center.

    Args:
        tensor (tf.Tensor): Feature tensor to be padded.
        final_size (int):
        dim (int): Dimension of tensor.

    Returns:
        padded_tensor (tf.Tensor): Padded feature tensor.
    """
    if dim == final_size:
        return tensor
    if final_size < dim:
        raise ValueError("Final size should be greater or equal than the current size!")

    padding_value = -1
    padded_tensor = tf.ones((tensor.shape[0], final_size, final_size))
    padded_tensor = padded_tensor * padding_value

    i = final_size // 2 - dim // 2

    padded_tensor = padded_tensor.numpy()
    padded_tensor[:, i : i + dim, i : i + dim] = tensor
    padded_tensor = tf.convert_to_tensor(padded_tensor)
    return padded_tensor
