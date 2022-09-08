import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    concatenate,
    GlobalMaxPool1D,
    Conv1D,
    ReLU,
    MaxPooling1D,
    Flatten,
)
from tensorflow.keras import backend as K, regularizers


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
    concat_relu = ReLU()(concat)

    # Path 1
    globalPooling = GlobalMaxPool1D()(concat_relu)
    # Path 2
    local_maxpool = MaxPooling1D(pool_size=4)(concat_relu)
    conv1 = Conv1D(filters=200, kernel_size=7)(local_maxpool)
    local_maxpool2 = MaxPooling1D(pool_size=4)(conv1)
    local_maxpool2 = Flatten()(local_maxpool2)

    concat2 = concatenate([globalPooling, local_maxpool2], axis=1)
    outputs = Dense(
        12,
        kernel_initializer="normal",
        kernel_regularizer=regularizers.l1(0.001),
        activation="linear",
    )(concat2)

    model = keras.Model(inputs=[fw_input, rc_input], outputs=outputs)

    model.summary()
    # keras.utils.plot_model(model, "MuSeAM_skip_connection.png")
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

    return model
