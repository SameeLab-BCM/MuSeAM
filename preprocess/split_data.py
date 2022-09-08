from .preprocess_data import preprocess
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle


def sharpr(dtype):
    # Load Train, Test, and Valid data
    train_seq = np.load("./data/sharpr/train_seqs.npy")
    train_readout = np.load("./data/sharpr/train_readout.npy")

    fwd_train = train_seq[::2, :, :]
    rc_train = train_seq[1::2, :, :]
    readout_train = train_readout[::2, :]

    fwd_test = np.load("./data/sharpr/test_seqs_fwd.npy")
    rc_test = np.load("./data/sharpr/test_seqs_rc.npy")
    readout_test = np.load("./data/sharpr/test_readout.npy")

    fwd_valid = np.load("./data/sharpr/valid_seqs_fwd.npy")
    rc_valid = np.load("./data/sharpr/valid_seqs_rc.npy")
    readout_valid = np.load("./data/sharpr/valid_readout.npy")

    if dtype == "two-inputs":
        return (
            fwd_train,
            rc_train,
            readout_train,
            fwd_test,
            rc_test,
            readout_test,
            fwd_valid,
            rc_valid,
            readout_valid,
        )

    if dtype == "one-input":
        return (
            train_seq,
            train_readout,
            fwd_test,
            readout_test,
            fwd_valid,
            readout_valid,
        )


def splitData(fasta_file, readout_file, partitionType, taskType=None):
    prep = preprocess(fasta_file, readout_file)
    dict = prep.one_hot_encode()

    fwd_fasta = dict["forward"]
    rc_fasta = dict["reverse"]
    readout = dict["readout"]

    if partitionType in ["10Fold", "fitAll"]:
        readout = np.array(readout)
        fwd_fasta, rc_fasta, readout = shuffle(
            fwd_fasta, rc_fasta, readout, random_state=0
        )
        return fwd_fasta, rc_fasta, readout

    if partitionType == "leaveOneOut":
        (
            fwd_train,
            fwd_test,
            rc_train,
            rc_test,
            readout_train,
            readout_test,
        ) = train_test_split(fwd_fasta, rc_fasta, readout, test_size=0.2, shuffle=True)

        readout_train = np.array(readout_train)
        readout_test = np.array(readout_test)

    return fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test
