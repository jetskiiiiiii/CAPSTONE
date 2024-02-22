A = get_one_hot_sequence("A")
U = get_one_hot_sequence("U")
G = get_one_hot_sequence("G")
C = get_one_hot_sequence("C")
allowed_pairs = [(A, U), (U, A), (G, C), (C, G), (G, U), (U, G)]
