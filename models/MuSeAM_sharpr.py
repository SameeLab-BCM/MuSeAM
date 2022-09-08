import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, concatenate, GlobalMaxPool1D, Conv1D
from tensorflow.keras import backend as K, regularizers
from scipy.stats import spearmanr


class ConvolutionLayer(Conv1D):
    def __init__(
        self,
        filters,
        kernel_size,
        data_format,
        padding="valid",
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        __name__="ConvolutionLayer",
        **kwargs
    ):
        super(ConvolutionLayer, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs
        )
        self.run_value = 1

    def call(self, inputs):
        if self.run_value > 2:

            x_tf = (
                self.kernel
            )  ##x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 100
            beta = 1 / alpha
            bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(
                lambda x: tf.math.scalar_mul(
                    beta,
                    tf.subtract(
                        tf.subtract(
                            tf.subtract(
                                tf.math.scalar_mul(alpha, x),
                                tf.expand_dims(
                                    tf.math.reduce_max(
                                        tf.math.scalar_mul(alpha, x), axis=1
                                    ),
                                    axis=1,
                                ),
                            ),
                            tf.expand_dims(
                                tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.subtract(
                                                tf.math.scalar_mul(alpha, x),
                                                tf.expand_dims(
                                                    tf.math.reduce_max(
                                                        tf.math.scalar_mul(alpha, x),
                                                        axis=1,
                                                    ),
                                                    axis=1,
                                                ),
                                            )
                                        ),
                                        axis=1,
                                    )
                                ),
                                axis=1,
                            ),
                        ),
                        tf.math.log(
                            tf.reshape(
                                tf.tile(bkg_tf, [tf.shape(x)[0]]),
                                [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                            )
                        ),
                    ),
                ),
                x_tf,
            )
            transf = tf.transpose(filt_list, [1, 2, 0])
            outputs = self._convolution_op(inputs, transf)
        else:
            outputs = self._convolution_op(inputs, self.kernel)

        self.run_value += 1
        return outputs


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

    customConv = ConvolutionLayer(
        filters=self.filters,
        kernel_size=self.kernel_size,
        data_format="channels_last",
        use_bias=True,
    )
    fw = customConv(fw_input)
    rc = customConv(rc_input)
    concat = concatenate([fw, rc], axis=1)
    globalPooling = GlobalMaxPool1D()(concat)

    fc1 = Dense(64)(globalPooling)
    fc2 = Dense(64)(globalPooling)
    fc3 = Dense(64)(globalPooling)
    fc4 = Dense(64)(globalPooling)
    fc5 = Dense(64)(globalPooling)
    fc6 = Dense(64)(globalPooling)
    fc7 = Dense(64)(globalPooling)
    fc8 = Dense(64)(globalPooling)
    fc9 = Dense(64)(globalPooling)
    fc10 = Dense(64)(globalPooling)
    fc11 = Dense(64)(globalPooling)
    fc12 = Dense(64)(globalPooling)

    out1 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc1)
    out2 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc2)
    out3 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc3)
    out4 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc4)
    out5 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc5)
    out6 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc6)
    out7 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc7)
    out8 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc8)
    out9 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc9)
    out10 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc10)
    out11 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc11)
    out12 = Dense(
        1,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(fc12)

    outputs = concatenate(
        [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12],
        axis=1,
    )

    model = keras.Model(inputs=[fw_input, rc_input], outputs=outputs)
    model.summary()
    # keras.utils.plot_model(model, "MuSeAM_sharpr.png")
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=[coeff_determination, spearman_fn],
    )

    return model
