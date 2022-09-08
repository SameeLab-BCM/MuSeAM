from saved_model import save_model
from preprocess.split_data import splitData, sharpr
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr

from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

import sklearn
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random

from utils.fetch_args import fetch_args
from utils.create_dirs import create_dirs

config = fetch_args()
create_dirs([config.outdir])

# Load models
import models.MuSeAM_classification as MuSeAM_classification
import models.MuSeAM_regression as MuSeAM_regression

# import models.MuSeAM_averagePooling as MuSeAM_averagePooling
# import models.MuSeAM_sharpr_single_input as MuSeAM_sharpr_single_input
# import models.MuSeAM_sharpr as MuSeAM_sharpr
# import models.sharpr as sharpr_model
# import models.MuSeAM_regression_pooling_layer as MuSeAM_regression_pooling_layer
# import models.MuSeAM_horizontal as MuSeAM_horizontal
# import models.MuSeAM_skip_connection as MuSeAM_skip_connection
# import models.MuSeAM_sumPooling as MuSeAM_sumPooling
# import models.MuSeAM_alpha as MuSeAM_alpha
# import models.MuSeAM_multiclass as MuSeAM_multiclass
# import models.deepsea as deepsea_model
# import models.deepsea_rnn as deepsea_rnn_model
# import models.basset as basset_model
# import models.basset_rnn as basset_rnn_model

# Reproducibility
seed = random.randint(1, 100000)
np.random.seed(seed)
tf.random.set_seed(seed)


class NN_model:
    """Neural network model training and evaluation class.

    Args:
        fasta_file (.fa): Sequence file in fasta format.
        readout_file (.dat): Expression data either regression or classification.
        filters (int): Number of filters to use in CNN.
        kernel_size (int): Size of the convolution kernel.
        epochs (int): Number of epochs.
        batch_size (int): Batch size in training.
        alpha (float32): Weight parameter in multinomial convolution.
        beta (float32): Inverse of alpha.
    """

    def __init__(
        self,
        fasta_file,
        readout_file,
        filters,
        kernel_size,
        epochs,
        batch_size,
        alpha,
        beta,
    ):
        self.fasta_file = fasta_file
        self.readout_file = readout_file
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.eval()
        # self.cross_val()
        # self.eval_sharpr()
        # self.load_weights()
        # self.fitAll()
        # self.pooling_coordinate()
        # self.pooling_layer()
        # self.relu_layer()
        # self.ablation_study()

    def load_weights(self):
        """Load saved weights from the model then perform post-hoc analysis."""
        reconstructed_model = keras.models.load_model(
            f"{config.outdir}/model", compile=False
        )

        # dtype = 'two-inputs'
        dtype = "one-input"
        if dtype == "two-inputs":
            (
                fwd_train,
                rc_train,
                readout_train,
                fwd_test,
                rc_test,
                readout_test,
                fwd_valid,
                rc_valid,
                readout_valid,
            ) = sharpr()
            seq_length = fwd_train.shape[1]
            model = MuSeAM_sharpr.create_model(self, seq_length)
            # print(reconstructed_model.layers[2].get_weights()[0].shape)
            filters = reconstructed_model.layers[2].get_weights()
            model.layers[2].set_weights(filters)

            model.fit(
                {"forward": fwd_train, "reverse": rc_train},
                readout_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            predictions = model.predict({"forward": fwd_test, "reverse": rc_test})
        if dtype == "one-input":
            (
                train_seq,
                train_readout,
                fwd_test,
                readout_test,
                fwd_valid,
                readout_valid,
            ) = sharpr(dtype)
            seq_length = fwd_train.shape[1]
            model = MuSeAM_sharpr_single_input.create_model(self, seq_length)
            filters = reconstructed_model.layers[2].get_weights()
            model.layers[1].set_weights(filters)

            model.fit(
                train_seq,
                train_readout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(fwd_valid, readout_valid),
            )
            predictions = model.predict(fwd_test)

        # Loop over each task
        spearmans = [
            spearmanr(readout_test[:, i], predictions[:, i])[0] for i in range(12)
        ]
        pearsons = [
            pearsonr(readout_test[:, i], predictions[:, i])[0] for i in range(12)
        ]

        # print(f'pearson averages are: {np.array(pearsons)[[2,5,8,11]]}')
        print(f"spearman averages are: {np.array(spearmans)[[2,5,8,11]]}")
        print(f"Mean spearman is {np.mean(spearmans)}")

    def eval(self):
        """One-fold training schema."""

        fwd_train, fwd_test, rc_train, rc_test, readout_train, readout_test = splitData(
            self.fasta_file, self.readout_file, partitionType="leaveOneOut"
        )
        seq_length = fwd_train.shape[1]

        # model = MuSeAM_classification.create_model(self, seq_length)
        # model = MuSeAM_multiclass.create_model(self, seq_length)
        model = MuSeAM_regression.create_model(self, seq_length)
        # model = MuSeAM_sumPooling.create_model(self, seq_length)

        callback = EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=3,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        model.fit(
            {"forward": fwd_train, "reverse": rc_train},
            readout_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.0,
            callbacks=[callback],
        )
        history = model.evaluate(
            {"forward": fwd_test, "reverse": rc_test}, readout_test
        )

        if config.task == "classification":
            pred_train = model.predict({"forward": fwd_train, "reverse": rc_train})
            pred_test = model.predict({"forward": fwd_test, "reverse": rc_test})
            pred_train = np.argmax(pred_train, axis=1)
            pred_test = np.argmax(pred_test, axis=1)

            trainAUC = sklearn.metrics.roc_auc_score(readout_train, pred_train)
            testAUC = sklearn.metrics.roc_auc_score(readout_test, pred_test)

            print("Seed number is {}".format(seed))
            print("Train-data AUC is ", trainAUC)
            print("Test-data AUC is ", testAUC)

        if config.task == "regression":
            print("Seed number is {}".format(seed))
            print("metric values of model.evaluate: " + str(history))
            print("metrics names are " + str(model.metrics_names))

            motif_weight = model.get_weights()
            dense_weight = motif_weight[2]
            np.savetxt(f"{config.outdir}/dense_weights_silencer.txt", dense_weight)
            save_model.save_model(self, model, alpha=120, path=f"{config.outdir}/model")

        preds = model.predict({"forward": fwd_test, "reverse": rc_test})
        plt.scatter(readout_test, preds, marker="o", s=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Scatter plot")
        plt.savefig(f"{config.outdir}/scatterplot_silencer.pdf")

    def fitAll(self):
        """Fitting the entire data into the trained model."""
        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, "fitAll"
        )
        seq_length = fwd_fasta.shape[1]

        if config.task == "regression":
            model = MuSeAM_regression.create_model(self, seq_length)
            # callback = EarlyStopping(monitor='spearman_fn', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            model.fit(
                {"forward": fwd_fasta, "reverse": rc_fasta},
                readout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.0,
            )
            history = model.evaluate(
                {"forward": fwd_fasta, "reverse": rc_fasta}, readout
            )
            save_model.save_model(self, model, alpha=120, path=f"{config.outdir}/model")

    def cross_val(self):
        """10-fold cross validation schema."""

        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, partitionType="10Fold"
        )
        seq_length = fwd_fasta.shape[1]

        # initialize metrics to save values
        trainAUCs = []
        testAUCs = []
        histories = []

        kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        ln = np.zeros(len(readout))

        for train, test in kFold.split(ln, ln):
            fwdTrain = fwd_fasta[train]
            fwdTest = fwd_fasta[test]
            rcTrain = rc_fasta[train]
            rcTest = rc_fasta[test]
            readoutTrain = readout[train]
            readoutTest = readout[test]

            if config.task == "classification":
                model = None
                model = MuSeAM_classification.create_model(self, seq_length)

                # Early stopping
                callback = EarlyStopping(
                    monitor="loss",
                    min_delta=0.001,
                    patience=3,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                )
                model.fit(
                    {"forward": fwdTrain, "reverse": rcTrain},
                    readoutTrain,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=0.0,
                    callbacks=[callback],
                )

                pred_train = model.predict({"forward": fwdTrain, "reverse": rcTrain})
                trainAUC = sklearn.metrics.roc_auc_score(readoutTrain, pred_train)
                trainAUCs.append(trainAUC)

                pred_test = model.predict({"forward": fwdTest, "reverse": rcTest})
                testAUC = sklearn.metrics.roc_auc_score(readoutTest, pred_test)
                testAUCs.append(testAUC)
            if config.task == "regression":
                model = None
                # model = MuSeAM_sumPooling.create_model(self, seq_length)
                # model = MuSeAM_alpha.create_model(self, seq_length)
                model = MuSeAM_regression.create_model(self, seq_length)
                # model = deepsea_model.create_model(self, seq_length)
                # model = deepsea_rnn_model.create_model(self, seq_length)

                # model.fit(fwdTrain, readoutTrain, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)
                # history = model.evaluate(fwdTest, readoutTest)

                # Early stopping
                # callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
                model.fit(
                    {"forward": fwdTrain, "reverse": rcTrain},
                    readoutTrain,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=0.0,
                )
                history = model.evaluate(
                    {"forward": fwdTest, "reverse": rcTest}, readoutTest
                )
                histories.append(history)

        print(f"Seed number is {seed}")

        if config.task == "classification":
            print(f"Train 10-fold AUCs are {trainAUCs}")
            print(f"Mean train AUC is {np.mean(trainAUCs)}")

            print(f"Test 10-fold AUCs are {testAUCs}")
            print(f"Mean test AUC is {np.mean(testAUCs)}")
        if config.task == "regression":
            g1 = []
            g2 = []
            g3 = []
            for i in histories:
                loss, r_2, spearman_val = i
                g1.append(loss)
                g2.append(r_2)
                g3.append(spearman_val)

            print("Mean loss of 10-fold cv is " + str(np.mean(g1)))
            print(g2)
            print("Mean R_2 score of 10-fold cv is " + str(np.mean(g2)))
            print(g3)
            print("Mean Spearman of 10-fold cv is " + str(np.mean(g3)))

    def pooling_layer(self):
        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, "fitAll"
        )
        seq_length = fwd_fasta.shape[1]
        model, model2, model3 = MuSeAM_regression_pooling_layer.create_model(
            self, seq_length
        )
        model.fit(
            {"forward": fwd_fasta, "reverse": rc_fasta},
            readout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.0,
        )
        pooling_output = model2.predict({"forward": fwd_fasta, "reverse": rc_fasta})
        np.savetxt(f"{config.outdir}/pooling_output.txt", pooling_output)

    def relu_layer(self):
        """Posthoc analysis on RELU layer output after training."""
        # Removed synthetic sequences
        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, "fitAll"
        )
        seq_length = fwd_fasta.shape[1]
        model, model2, model3 = MuSeAM_regression_pooling_layer.create_model(
            self, seq_length
        )

        # callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        model.fit(
            {"forward": fwd_fasta, "reverse": rc_fasta},
            readout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.0,
        )

        relu_output = model3.predict({"forward": fwd_fasta, "reverse": rc_fasta})
        print(relu_output.shape)
        np.save(f"{config.outdir}/RELU_output", relu_output)

    def pooling_coordinate(self):
        """Get MaxPool location for each convolution."""
        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, "fitAll"
        )

        # Get maxpool hitting coordinates
        os.makedirs(f"{config.outdir}/maxpool_index")
        dir = f"{config.outdir}/motif_files/"

        files = []
        for file in os.listdir(dir):
            if file.startswith("filter"):
                files.append(dir + file)
        files = sorted(files)

        comb_filter = np.loadtxt(files[0])
        for i in files[1:]:  # length of 512
            conv_filter = np.loadtxt(i)  # shape (12,4)
            comb_filter = np.concatenate(
                (comb_filter, conv_filter), axis=0
            )  # shape of (512*12=6144, 4)

        for i in range(len(fwd_fasta)):
            seq1 = fwd_fasta[i]  ## length of sequence is 171
            seq2 = rc_fasta[i]

            start_position = []
            for j in range(self.filters):
                vals_fw = []
                vals_rc = []
                # sequence length after convolution
                seq_length = len(seq1) - self.kernel_size + 1
                for k in range(seq_length):
                    fw_one_hot = seq1[k : k + self.kernel_size, :]
                    rc_one_hot = seq2[k : k + self.kernel_size, :]
                    # sum the all convolved values((171,4) * (12,4))
                    val = np.multiply(
                        fw_one_hot,
                        comb_filter[
                            self.kernel_size * j : self.kernel_size * j
                            + self.kernel_size,
                            :,
                        ],
                    )

                    val1 = np.sum(
                        np.multiply(
                            fw_one_hot,
                            comb_filter[
                                self.kernel_size * j : self.kernel_size * j
                                + self.kernel_size,
                                :,
                            ],
                        )
                    )

                    val2 = np.sum(
                        np.multiply(
                            rc_one_hot,
                            comb_filter[
                                self.kernel_size * j : self.kernel_size * j
                                + self.kernel_size,
                                :,
                            ],
                        )
                    )
                    vals_fw.append(val1)
                    vals_rc.append(val2)

                vals = vals_fw + vals_rc

                # get the maximum of 320 values
                max_val = max(vals)
                # get the indices for all maximum hits for each filter
                ind = vals.index(max_val)

                if ind >= seq_length:
                    ind2 = (
                        ind - seq_length
                    )  ## for reverse complement, subtract 160 to match the location
                    ind3 = seq_length - 1 - ind2  # now we transform it: 0 --> n-ind-1
                    ind = int(ind3)
                start_position.append(ind)

            end_position = np.add(self.kernel_size - 1, start_position)

            # reshape in order to concatenate... originally (512,) --> (512,1)
            start_position = np.reshape(start_position, (self.filters, 1))
            end_position = np.reshape(end_position, (self.filters, 1))

            concat_position = np.concatenate((start_position, end_position), axis=1)

            np.savetxt(
                f'.{config.outdir}/maxpool_index/maxpool_{"{0:0=3d}".format(i)}.txt',
                concat_position,
            )

    def ablation_study(self):
        """Ablation study by removing one convolution at a time."""
        # first train the model
        fwd_fasta, rc_fasta, readout = splitData(
            self.fasta_file, self.readout_file, "fitAll"
        )
        seq_length = fwd_fasta.shape[1]

        if config.task == "regression":
            perfs = []  # performances
            model = MuSeAM_regression.create_model(self, seq_length)
            # callback = EarlyStopping(monitor='spearman_fn', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            model.fit(
                {"forward": fwd_fasta, "reverse": rc_fasta},
                readout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.0,
            )

            history = model.evaluate(
                {"forward": fwd_fasta, "reverse": rc_fasta}, readout
            )
            baseline = history[2]
            motif_weight = model.get_weights()
            W = motif_weight[0].copy()

            for i in range(W.shape[2]):
                new_weight = W.copy()
                new_weight[:, :, i] = np.zeros((12, 4))

                # Loop over by setting each filter with zeros and predicting with new set of features
                motif_weight[0] = new_weight
                model.set_weights(motif_weight)

                history = model.evaluate(
                    {"forward": fwd_fasta, "reverse": rc_fasta}, readout
                )
                perf = history[-1]  # Spearman rank correlation

                # Save performance change (delta)
                delta = baseline - perf
                perfs.append(delta)
            np.savetxt(f"{config.outdir}/ablation.txt", perfs)

    def eval_sharpr(self):
        """Evaluation for DragoNN sharpr model."""
        # dtype = 'two-inputs'
        dtype = "one-input"

        if dtype == "two-inputs":
            (
                fwd_train,
                rc_train,
                readout_train,
                fwd_test,
                rc_test,
                readout_test,
                fwd_valid,
                rc_valid,
                readout_valid,
            ) = sharpr(dtype)
            seq_length = fwd_train.shape[1]
            # model = MuSeAM_sharpr.create_model(self, seq_length)
            # model = MuSeAM_skip_connection.create_model(self, seq_length)
            # model = MuSeAM_horizontal.create_model(self, seq_length)
            model = MuSeAM_averagePooling.create_model(self, seq_length)
            # model = sharpr_model.create_model(self, seq_length)

            model.fit(
                {"forward": fwd_train, "reverse": rc_train},
                readout_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(
                    {"forward": fwd_valid, "reverse": rc_valid},
                    readout_valid,
                ),
            )
            predictions = model.predict({"forward": fwd_test, "reverse": rc_test})
            predictions_valid = model.predict(
                {"forward": fwd_valid, "reverse": rc_valid}
            )

        if dtype == "one-input":
            (
                train_seq,
                train_readout,
                fwd_test,
                readout_test,
                fwd_valid,
                readout_valid,
            ) = sharpr(dtype)
            seq_length = fwd_train.shape[1]
            model = MuSeAM_sharpr_single_input.create_model(self)

            model.fit(
                train_seq,
                train_readout,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(fwd_valid, readout_valid),
            )
            predictions = model.predict(fwd_test)
            predictions_valid = model.predict(fwd_valid)

        # Loop over each task
        spearmans = [
            spearmanr(readout_test[:, i], predictions[:, i])[0] for i in range(12)
        ]
        pearsons = [
            pearsonr(readout_test[:, i], predictions[:, i])[0] for i in range(12)
        ]

        spearmans_valid = [
            spearmanr(readout_valid[:, i], predictions_valid[:, i])[0]
            for i in range(12)
        ]
        pearsons_valid = [
            pearsonr(readout_valid[:, i], predictions_valid[:, i])[0] for i in range(12)
        ]

        print(f"Test spearman averages are: {np.array(spearmans)[[2,5,8,11]]}")
        print(f"Test Mean spearman is {np.mean(spearmans)}")

        print(
            f"Validation spearman averages are: {np.array(spearmans_valid)[[2,5,8,11]]}"
        )
        print(f"Validation Mean spearman is {np.mean(spearmans_valid)}")
