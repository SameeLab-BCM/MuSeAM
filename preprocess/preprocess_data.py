import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from .dinucleotide import mono_to_dinucleotide, dinucleotide_one_hot_encode


class preprocess:
    def __init__(self, fasta_file, readout_file):
        self.fasta_file = fasta_file
        self.readout_file = readout_file

    def read_fasta_into_list(self):
        all_seqs = []
        with open(self.fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                line = line.upper()
                if line[0] == ">":
                    continue
                else:
                    line = line.replace("N", "")
                    all_seqs.append(line)
        return all_seqs

    # augment the samples with reverse complement
    def rc_comp2(self):
        def rc_comp(seq):
            rc_dict = {"A": "T", "C": "G", "G": "C", "T": "A"}
            rc_seq = "".join([rc_dict[c] for c in seq[::-1]])
            return rc_seq

        seqn = self.read_fasta_into_list()
        all_sequences = []
        for seq in range(len(seqn)):
            all_sequences.append(rc_comp(seqn[seq]))

        # return all_sequences

        return all_sequences

    # to augment on readout data
    def read_readout(self):
        all_readout = []
        with open(self.readout_file, "r") as f:
            all_readout = list(map(float, f.readlines()))

        return all_readout

    def augment(self):
        new_fasta = self.read_fasta_into_list()
        rc_fasta = self.rc_comp2()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout": readout, "rc_fasta": rc_fasta}
        return dict

    def without_augment(self):
        new_fasta = self.read_fasta_into_list()
        readout = self.read_readout()

        dict = {"new_fasta": new_fasta, "readout": readout}
        return dict

    def one_hot_encode(self):
        integer_encoder = LabelEncoder()
        one_hot_encoder = OneHotEncoder(categories="auto")

        # dict = self.without_augment()
        dict = self.augment()

        forward = []
        reverse = []

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in dict["new_fasta"]:
            new_seq = "ACGT" + sequence
            temp_seqs.append(new_seq)

        for sequence in temp_seqs:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            forward.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        lengths = []
        for i in range(len(forward)):
            length = len(forward[i])
            lengths.append(length)
        max_length = max(lengths)  # get the maxmimum length of all sequences

        for i in range(len(forward)):
            while len(forward[i]) < max_length:
                forward[i] = np.vstack((forward[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        features = []
        for sequence in forward:
            new = sequence[4:]
            features.append(new)

        features = np.stack(features)
        dict["forward"] = features

        # some sequences do not have entire 'ACGT'
        temp_seqs = []
        for sequence in dict["rc_fasta"]:
            new_seq = "ACGT" + sequence
            temp_seqs.append(new_seq)

        for sequence in temp_seqs:
            integer_encoded = integer_encoder.fit_transform(list(sequence))
            integer_encoded = np.array(integer_encoded).reshape(-1, 1)
            one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
            reverse.append(one_hot_encoded.toarray())

        # padding [0,0,0,0] such that sequences have same length
        lengths = []
        for i in range(len(reverse)):
            length = len(reverse[i])
            lengths.append(length)
        max_length = max(lengths)  # get the maxmimum length of all sequences

        for i in range(len(reverse)):
            while len(reverse[i]) < max_length:
                reverse[i] = np.vstack((reverse[i], [0, 0, 0, 0]))

        # remove first 4 nucleotides
        features = []
        for sequence in reverse:
            new = sequence[4:]
            features.append(new)

        features = np.stack(features)
        dict["reverse"] = features

        return dict

    def dinucleotide_encode(self):
        new_fasta = self.read_fasta_into_list()
        sequences = mono_to_dinucleotide(new_fasta)

        input_features = dinucleotide_one_hot_encode(sequences)

        dict = self.without_augment()
        dict["input_features"] = input_features
        return dict
