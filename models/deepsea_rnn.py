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
from tensorflow.keras.layers import LSTM


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

    model = Sequential()

    model.add(Conv1D(filters=320, kernel_size=8, input_shape=(seq_length, 4)))
    model.add(MaxPooling1D(pool_size=4, strides=4))
    model.add(Dropout(rate=0.20))

    model.add(Conv1D(filters=480, kernel_size=8))
    model.add(MaxPooling1D(pool_size=4, strides=4))
    model.add(Dropout(rate=0.20))

    model.add(Conv1D(filters=960, kernel_size=8))
    model.add(Dropout(rate=0.50))

    model.add(LSTM(units=1000, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(925, activation="relu"))
    # Output Layer
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[coeff_determination, spearman_fn],
    )
    return model
