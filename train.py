from main import NN_model
import time
import sys

from utils.fetch_args import fetch_args

config = fetch_args()

# get dictionary from text file
def train(file_name):
    dict = {}
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split()
            dict[key] = val

    # change string values to integer values
    dict["filters"] = int(dict["filters"])
    dict["kernel_size"] = int(dict["kernel_size"])
    dict["epochs"] = int(dict["epochs"])
    dict["batch_size"] = int(dict["batch_size"])
    dict["alpha"] = int(dict["alpha"])
    dict["beta"] = float(dict["beta"])

    return dict


def main():
    ## excute the code
    start_time = time.time()
    dict = train(config.param)

    NN_model(
        config.input,
        config.output,
        filters=dict["filters"],
        kernel_size=dict["kernel_size"],
        epochs=dict["epochs"],
        batch_size=dict["batch_size"],
        alpha=dict["alpha"],
        beta=dict["beta"],
    )

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    sys.exit(main())
