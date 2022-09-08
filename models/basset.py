from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling2D,
    Dropout,
    Flatten,
    BatchNormalization,
    MaxPooling1D,
)
from tensorflow.keras import backend as K, regularizers
from scipy.stats import spearmanr, pearsonr


def create_model(self, seq_length):
    def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    def spearman_fn(y_true, y_pred):
        return tf.py_function(
            spearmanr,
            [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
            Tout=tf.float32,
        )

    model = model = Sequential()

    # First Conv1D
    model.add(Conv1D(filters=300, kernel_size=19, input_shape=(seq_length, 4)))

    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=3))

    model.add(Conv1D(filters=200, kernel_size=11))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters=200, kernel_size=7))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(rate=0.30))
    model.add(Dense(164, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[coeff_determination, spearman_fn],
    )
    return model
