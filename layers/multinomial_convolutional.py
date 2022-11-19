import tensorflow as tf
from tensorflow.keras.layers import Conv1D


class MultinomialConvolutionLayer(Conv1D):
    def __init__(
        self,
        filters,
        alpha,
        beta,
        kernel_size,
        background=[0.295, 0.205, 0.205, 0.295],
        padding="valid",
        data_format="channels_last",
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        __name__="MultinomialConvolutionLayer",
        **kwargs
    ):
        super(MultinomialConvolutionLayer, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            data_format=data_format,
            kernel_initializer=kernel_initializer,
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self.run_value = 1
        self.bkg = tf.cast(tf.constant(background), tf.float32)

    def call(self, inputs):

        if self.run_value > 2:

            x_tf = tf.transpose(self.kernel, [2, 0, 1])

            filt_list = tf.map_fn(
                lambda x: tf.math.scalar_mul(
                    self.beta,
                    tf.subtract(
                        tf.subtract(
                            tf.subtract(
                                tf.math.scalar_mul(self.alpha, x),
                                tf.expand_dims(
                                    tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1),
                                    axis=1,
                                ),
                            ),
                            tf.expand_dims(
                                tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.subtract(
                                                tf.math.scalar_mul(self.alpha, x),
                                                tf.expand_dims(
                                                    tf.math.reduce_max(
                                                        tf.math.scalar_mul(self.alpha, x),
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
                                tf.tile(self.bkg, [tf.shape(x)[0]]),
                                [tf.shape(x)[0], tf.shape(self.bkg)[0]],
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
