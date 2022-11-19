import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate, GlobalMaxPool1D, ReLU
from tensorflow.keras import backend as K, regularizers
from tensorflow import keras
from scipy.stats import spearmanr

from layers import MultinomialConvolutionLayer


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

    fw_input = keras.Input(shape=(seq_length, 4), name="forward")
    rc_input = keras.Input(shape=(seq_length, 4), name="reverse")

    customConv = MultinomialConvolutionLayer(
        filters=self.filters,
        kernel_size=self.kernel_size,
        use_bias=True,
        alpha=self.alpha,
        beta=self.beta,
    )
    fw = customConv(fw_input)
    rc = customConv(rc_input)
    concat = concatenate([fw, rc], axis=1)

    activation = ReLU()(concat)

    globalPooling = GlobalMaxPool1D()(activation)
    outputs = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(globalPooling)

    model = keras.Model(inputs=[fw_input, rc_input], outputs=outputs)
    model.summary()

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[coeff_determination, spearman_fn],
    )

    return model
