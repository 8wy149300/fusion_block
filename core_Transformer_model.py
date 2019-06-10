# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer


# from tensorflow.python.layers import core as core_layer
# import numpy as np
# from hyper_and_conf import hyper_beam_search as beam_search


class LinearEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            max_seq_len,
            vocabulary_size,
            embedding_size=512,
            batch_size=64,
            num_units=512,
            num_heads=6,
            num_encoder_layers=6,
            dropout=0.4,
            eos_id=1,
            pad_id=0,
    ):
        super(LinearEncoder, self).__init__(name="linear_encoder")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.eos_id = eos_id
        self.pad_id = pad_id

    def build(self, input_shape):
        self.stacked_encoder = []
        for i in range(self.num_encoder_layers):
            self_attention = hyper_layer.SelfAttention(
                num_heads=self.num_heads,
                num_units=self.num_units,
                dropout=self.dropout,
            )
            ffn = hyper_layer.Feed_Forward_Network(
                num_units=4 * self.num_units, dropout=self.dropout)
            self.stacked_encoder.append([
                hyper_layer.NormBlock(self_attention, self.dropout),
                hyper_layer.NormBlock(ffn, self.dropout),
            ])
        self.encoder_output = hyper_layer.LayerNorm()
        super(LinearEncoder, self).build(input_shape)

    def call(self, inputs, attention_bias, training):
        # inputs= inputs
        # inputs = Q
        with tf.name_scope("stacked_encoder"):
            for index, layer in enumerate(self.stacked_encoder):
                with tf.name_scope("layer_%d" % index):
                    self_att = layer[0]
                    ffn = layer[1]
                    inputs = self_att(
                        inputs, attention_bias, training=training)
                    inputs = ffn(inputs, training=training)
        return self.encoder_output(inputs)

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c


class LinearDecoder(tf.keras.layers.Layer):
    def __init__(
            self,
            max_seq_len,
            vocabulary_size,
            embedding_size=512,
            batch_size=64,
            num_units=512,
            num_heads=6,
            num_decoder_layers=6,
            dropout=0.4,
            eos_id=1,
            pad_id=0,
    ):
        super(LinearDecoder, self).__init__(name="Linear_decoder")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.eos_id = eos_id
        self.pad_id = pad_id

    def build(self, input_shape):

        self.decoder_output = hyper_layer.LayerNorm()
        self.stacked_decoder = []
        for i in range(self.num_decoder_layers):
            self_attention = hyper_layer.SelfAttention(
                num_heads=self.num_heads,
                num_units=self.num_units,
                dropout=self.dropout,
            )

            attention = hyper_layer.Attention(
                num_heads=self.num_heads,
                num_units=self.num_units,
                dropout=self.dropout,
            )
            ffn = hyper_layer.Feed_Forward_Network(
                num_units=4 * self.num_units, dropout=self.dropout)
            self.stacked_decoder.append([
                hyper_layer.NormBlock(self_attention, self.dropout),
                hyper_layer.NormBlock(attention, self.dropout),
                hyper_layer.NormBlock(ffn, self.dropout),
            ])
            super(LinearDecoder, self).build(input_shape)

    def call(
            self,
            inputs,
            enc,
            decoder_self_attention_bias,
            attention_bias,
            cache=None,
            training=False,
    ):
        with tf.name_scope("stacked_decoder"):
            for index, layer in enumerate(self.stacked_decoder):
                self_att = layer[0]
                att = layer[1]
                ffn = layer[2]
                layer_name = "layer_%d" % index
                layer_cache = cache[layer_name] if cache is not None else None
                with tf.name_scope("layer_%d" % index):
                    inputs = self_att(
                        inputs,
                        decoder_self_attention_bias,
                        training=training,
                        cache=layer_cache,
                    )
                    inputs = att(
                        inputs, enc, attention_bias, training=training)
                    inputs = ffn(inputs, training=training)
        return self.decoder_output(inputs)

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout": self.dropout,
        }
        return c
