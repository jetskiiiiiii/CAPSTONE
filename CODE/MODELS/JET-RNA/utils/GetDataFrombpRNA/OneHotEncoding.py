import numpy as np

__all__ = ["get_one_hot_encoding"]


# one hot sequence
def get_one_hot_sequence(sequence):
    bases = np.array(["A", "U", "G", "C"])
    encoding = []
    for i in range(len(sequence)):
        if i == "P":
            encoding[i].append([0, 0, 0, 0])
        encoding.append([])
        for j in range(len(bases)):
            if sequence[i] == bases[j]:
                encoding[i].append(1)
            else:
                encoding[i].append(0)
    return np.array(encoding)


# SPOT-RNA one-hot encoding
def one_hot(seq):
    RNN_seq = seq
    BASES = "AUCG"
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [
            (
                [(bases == base.upper()).astype(int)]
                if str(base).upper() in BASES
                else np.array([[-1] * len(BASES)])
            )
            for base in RNN_seq
        ]
    )

    return feat


# GCNFOLD one-hot encoding
def matrix2seq(one_hot_matrices):
    d = {
        "A": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "U": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "C": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        "G": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }
    seq_list = []
    for i in range(one_hot_matrices.shape[0]):
        one_hot_matrice = one_hot_matrices[i, 0, :]
        seq = ""
        for loc in range(one_hot_matrice.shape[0]):
            if one_hot_matrice[loc, 0] == 1:
                seq += "A"
            elif one_hot_matrice[loc, 1] == 1:
                seq += "G"
            elif one_hot_matrice[loc, 2] == 1:
                seq += "C"
            elif one_hot_matrice[loc, 3] == 1:
                seq += "U"
            else:
                seq += "N"
        seq_list.append(seq)

    return seq_list
