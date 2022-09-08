import argparse
import datetime


def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="MuSeAM_regression",
        help="Define the model to use",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="regression",
        help="Task type: classification or regression",
    )

    parser.add_argument(
        "-ip",
        "--input",
        type=str,
        default="./data/liver_enhancer/sequences.fa",
        help="Input data",
    )
    parser.add_argument(
        "-op",
        "--output",
        type=str,
        default="./data/liver_enhancer/wt_readout.dat",
        help="Input data",
    )
    parser.add_argument(
        "-p", "--param", type=str, default="parameters.txt", help="Parameter config"
    )

    args = parser.parse_args()

    currentTime = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    args.outdir = f"./output/{currentTime}"

    return args
