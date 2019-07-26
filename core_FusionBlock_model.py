# encoding=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer

from hyper_and_conf import conf_fn
from tensorflow.python.keras import regularizers
L2_WEIGHT_DECAY = 1e-4


class Fusion_Block(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, name="fusion_block"):
        super(Fusion_Block, self).__init__(name="fusion_block")
        self.num_units = num_units
        self.dropout = dropout

    def build(self, input_shape):
        # self.concatenation_norm = hyper_layer.LayerNorm()

        # self.src_1_kernel = tf.keras.layers.Dense(self.num_units, name="src_1")
        #
        # self.shared_pre_kernel = tf.keras.layers.Conv1D(
        #     self.num_units, 3, use_bias=False, padding='same')

        self.fusion_kernel = tf.keras.layers.Dense(
            self.num_units, name="fusion_kernel", use_bias=False)
        # self.fusion_kernel = hyper_layer.ResnetIdentityBlock(
        #     1, (self.num_units, int(self.num_units / 16), self.num_units))
        # self.pre_mask_id = tf.keras.layers.Conv1D(
        #     self.num_units, 5, use_bias=False, padding='same')
        self.FFN_scale = tf.keras.layers.Dense(
            self.num_units * 4, name="FFN_scale_kernel", use_bias=False)
        # self.FFN_dense = hyper_layer.ResnetIdentityBlock(
        #     1, (self.num_units * 2, self.num_units * 8, self.num_units * 2))
        self.FFN_norm = tf.keras.layers.BatchNormalization()
        self.FFN_dense = tf.keras.layers.Dense(
            self.num_units * 2, name="FFN_scale_kernel", use_bias=False)
        # self.FFN_dense = hyper_layer.ResnetIdentityBlock(
        #     1, (self.num_units * 2, int(self.num_units / 16) * 2,
        #         self.num_units * 2))
        # self.fusion_kernel = tf.keras.layers.Conv1D(
        #     self.num_units, 5, use_bias=False, padding='same')
        # self.FFN_dense = tf.keras.layers.Conv1D(
        #     self.num_units * 2,
        #     1,
        #     use_bias=False,
        #     kernel_initializer='he_normal',
        #     kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        #     padding='same')
        # self.fusion_kernel = tf.keras.layers.Conv1D(
        #     self.num_units,
        #     1,
        #     use_bias=False,
        #     kernel_initializer='he_normal',
        #     kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        #     padding='same')
        # self.res_norm = hyper_layer.LayerNorm()
        # self.position_projection = tf.keras.layers.Dense(
        #     self.num_units * 2, use_bias=False, name="res_projection")
        # self.FFN_pre_norm = hyper_layer.LayerNorm()
        self.FFN_post_norm = tf.keras.layers.BatchNormalization()
        self.output_norm = tf.keras.layers.BatchNormalization()

        super(Fusion_Block, self).build(input_shape)

    def call(self, inputs, padding, training=False):
        src_1, src_2 = inputs

        length = tf.shape(input=src_1)[1]
        # positional_input = conf_fn.get_position_encoding(
        #     length, self.num_units * 2)
        # src_1 = src_1 + positional_input
        # src_2 = src_2 + positional_input
        # src_1 = self.shared_pre_kernel(src_1)
        # src_2 = self.shared_pre_kernel(src_2)

        padding = self.padding((1 - padding))
        org = tf.concat([src_1, src_2], axis=-1)
        # org = (org + positional_input) * padding
        # FNN block
        outputs = org
        outputs = tf.nn.relu(self.FFN_norm(self.FFN_scale(outputs))) * padding
        if training:
            outputs = tf.nn.dropout(outputs, rate=self.dropout)
        outputs = tf.nn.relu(self.FFN_dense(outputs)) * padding
        outputs = self.FFN_post_norm(outputs) * padding
        # For short cut connection
        # org = self.inputs_projection(outputs)
        # res = self.res_projection(self.fusion_kernel(outputs))

        # res = self.output_norm(outputs) * padding
        res = tf.nn.relu(outputs) *padding

        if training:
            outputs = tf.nn.dropout(outputs, rate=self.dropout)
        res = self.fusion_kernel(res) * padding
        # if training:
        #     res = tf.nn.dropout(res, rate=self.dropout)
        res = self.output_norm(res) * padding
        # back to batch major
        return tf.nn.relu(res) * padding
        # return tf.transpose(res, [1, 0, 2])

    def padding(self, padding):
        padding = tf.expand_dims(padding, axis=-1)
        return padding

    def get_config(self):
        # config = super(Fusion_Block, self).get_config()
        c = {"num_units": self.num_units, "dropout": self.dropout}
        # config.update(c)
        return c
