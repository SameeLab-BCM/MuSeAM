import sys
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling2D,
    Dropout,
    Flatten,
    BatchNormalization,
)


def create_model(self, seq_length):
    model = Sequential()

    model.add(Conv1D(120, 5, activation="relu", input_shape=(seq_length, 4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv1D(120, 5, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv1D(120, 5, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(12, activation="linear"))
    model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
