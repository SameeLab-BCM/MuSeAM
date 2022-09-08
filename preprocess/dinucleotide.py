import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def mono_to_dinucleotide(seqs):

    dinucleotides = {
        "AA": "A",
        "AC": "B",
        "AG": "C",
        "AT": "D",
        "CA": "E",
        "CC": "F",
        "CG": "G",
        "CT": "H",
        "GA": "I",
        "GC": "J",
        "GG": "K",
        "GT": "L",
        "TA": "M",
        "TC": "O",
        "TG": "P",
        "TT": "Q",
    }

    dinucleotide_seqs = []
    for s in seqs:
        s = s.upper()
        t = ""
        n = len(s)
        # assert n > 1
        for i in range(n - 1):
            cur_dinucleotide = s[i : i + 2]
            t += dinucleotides[cur_dinucleotide]
        dinucleotide_seqs.append(t)

    return dinucleotide_seqs


"""
l = ['ACCGTGACA', 'TGTCATTTTA']
m = mono_to_dinucleotide(l)
for a, b in zip(l, m):
	print(a, b)
"""


def dinucleotide_one_hot_encode(seqs):

    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix where
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories="auto")

    temp_seqs = []
    for sequence in seqs:
        new_seq = "ABCDEFGHIJKLMOPQ" + sequence
        temp_seqs.append(new_seq)

    input_features = []
    for sequence in temp_seqs:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    features = []
    for sequence in input_features:
        new = sequence[16:]
        features.append(new)

    input_features = np.stack(features)

    return input_features
