# encoding=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer

from hyper_and_conf import conf_fn


class Fusion_Block(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, name="fusion_block"):
        super(Fusion_Block, self).__init__(name="fusion_block")
        self.num_units = num_units
        self.dropout = dropout

    def build(self, input_shape):
        self.concatenation_norm = hyper_layer.LayerNorm()

        self.fusion_kernel = tf.keras.layers.Dense(
            self.num_units, use_bias=True, activation=tf.nn.relu, name="fusion_kernel"
        )
        # self.inputs_projection = tf.keras.layers.Conv1D(
        #     self.num_units,
        #     1,
        #     use_bias=False,
        #     kernel_initializer='he_normal',
        # )
        self.inputs_projection = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="projection"
        )
        self.FFN = hyper_layer.Feed_Forward_Network(self.num_units * 8, self.dropout)
        self.FNN_block = hyper_layer.NormBlock(self.FFN, self.dropout)
        self.FFN_post_norm = hyper_layer.LayerNorm()
        self.res_norm = hyper_layer.LayerNorm()
        self.res_projection = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="res_projection"
        )
        self.output_norm = hyper_layer.LayerNorm()
        super(Fusion_Block, self).build(input_shape)

    def call(self, inputs, training=False):
        src_1, src_2 = inputs
        length = tf.shape(input=src_1)[1]
        img_input_padding = tf.cast(
            tf.not_equal(tf.reduce_sum(src_1, -1), 0.0), dtype=tf.float32
        )
        positional_input = conf_fn.get_position_encoding(length, self.num_units * 2)
        padding = self.padding(img_input_padding)
        org = tf.concat([src_1, src_2], axis=-1)
        org = (org + positional_input) * padding
        org = self.concatenation_norm(org)
        # we use time major here
        org = tf.transpose(org, [1, 0, 2])
        # FNN block
        outputs = self.FNN_block(org, training=training)
        outputs = self.FFN_post_norm(outputs)
        # For short cut connection
        org = self.inputs_projection(outputs)
        res = self.res_projection(self.fusion_kernel(outputs))
        if training:
            res = tf.nn.dropout(res, rate=self.dropout)
        res = self.output_norm(res + org)
        # back to batch major
        return tf.transpose(res, [1, 0, 2])

    def padding(self, padding):
        padding = tf.expand_dims(padding, axis=-1)
        return padding

    def get_config(self):
        # config = super(Fusion_Block, self).get_config()
        c = {"num_units": self.num_units, "dropout": self.dropout}
        # config.update(c)
        return c
