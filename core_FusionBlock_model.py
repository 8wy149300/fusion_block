# encoding=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer


class Fusion_Block(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, name="fusion_block"):
        super(Fusion_Block, self).__init__(name="fusion_block")
        self.num_units = num_units
        self.dropout = dropout
        self.FFN = hyper_layer.Feed_Forward_Network(
            num_units=self.num_units * 4, dropout=dropout)
        self.position_encoding = hyper_layer.Positional_Encoding(
            2 * self.num_units)
        self.norm = hyper_layer.LayerNorm()
        self.fusion_kernel = self.add_variable(
            name='fusion_kernel',
            shape=[2 * self.num_units, self.num_units],
            initializer=tf.keras.initializers.get("glorot_uniform"))
        self.fusion_bias = self.add_variable(
            shape=self.num_units,
            name="fusion_bias",
            initializer=tf.keras.initializers.get('zeros'))

    def call(self, inputs, training=False):
        src_1, src_2 = inputs
        length = tf.shape(input=src_1)[1]
        org = tf.keras.layers.concatenate([src_1, src_2], axis=-1)
        positional_input = self.position_encoding(length)
        outputs = org + positional_input
        if training is not False:
            dropout_mask_inputs = tf.keras.backend.dropout(
                tf.ones_like(outputs), self.dropout)
            outputs = outputs * dropout_mask_inputs
        # norm = self.norm(outputs)
        res = self.FFN(outputs, training=training)
        if training is not False:
            dropout_mask_inputs = tf.keras.backend.dropout(
                tf.ones_like(res), self.dropout)
            res = res * dropout_mask_inputs
        res = tf.keras.backend.dot(res, self.fusion_kernel)
        res = self.norm(res)
        res = res + self.fusion_bias
        res = tf.keras.layers.Activation("relu")(res)
        # res = tf.keras.layers.Activation("relu")(res)
        return res

    def get_config(self):
        config = super(Fusion_Block, self).get_config()
        c = {'num_units': self.num_units, 'dropout': self.dropout}
        config.update(c)
        return config
