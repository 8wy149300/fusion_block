# encoding=utf8
import tensorflow as tf
from core_resTransformer import ResLinearEncoder, LinearDecoder
from core_Transformer_model import LinearEncoder
from hyper_and_conf import hyper_layer
from hyper_and_conf import hyper_beam_search as beam_search
import numpy as np
# from hyper_and_conf import hyper_param
# from tensorflow.python.keras import initializers
from hyper_and_conf import hyper_util, conf_fn
# from tensorflow.python.keras import regularizers
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

GPU = conf_fn.get_available_gpus() if conf_fn.get_available_gpus() > 0 else 1

# PADDED_IMG = 150
# PADDED_TEXT = 80
PADDED_IMG = 50
PADDED_TEXT = 1

FEATURE_IMG = 32 * 64 * 3

# class Shared_Projection(tf.keras.layers.Layer):
#     def __init__(self, num_units):
#         super(Shared_Projection, self).__init__()
#         self.num_units = num_units
#         self.shared_projection = self.add_weight(
#             shape=[self.num_units, self.num_units],
#             dtype="float32",
#             name="shared",
#             initializer=tf.random_normal_initializer(
#                 mean=0., stddev=self.num_units**-0.5))
#
#     def build(self, input_shape):
#         super(Shared_Projection, self).build(input_shape)
#
#     def call(self, inputs, training, transpose=False):
#         length = tf.shape(inputs)[1]
#         inputs = tf.reshape(inputs, [-1, self.num_units])
#         inputs = tf.matmul(
#             inputs, self.shared_projection, transpose_b=transpose)
#         inputs = tf.reshape(inputs, [-1, length, self.num_units])
#         return inputs
#
#     def get_config(self):
#         return {"num_units": self.num_units}
#
#
# class IMG_POST(tf.keras.layers.Layer):
#     def __init__(self, num_units, dropout):
#         self.num_units = num_units
#         self.dropout = dropout
#         super(IMG_POST, self).__init__()
#
#     def build(self, input_shape):
#         self.extra_norm = tf.keras.layers.BatchNormalization(
#             axis=-1,
#             momentum=BATCH_NORM_DECAY,
#             epsilon=BATCH_NORM_EPSILON,
#         )
#         self.out_weights = self.add_weight(
#             shape=[1024, self.num_units],
#             dtype="float32",
#             name="out_weights",
#             initializer=tf.random_normal_initializer(
#                 mean=0., stddev=self.num_units**-0.5))
#
#         super(IMG_POST, self).build(input_shape)
#
#     def call(self, inputs, padding=None, training=False):
#
#         inputs = self.extra_norm(inputs)
#         inputs = tf.matmul(inputs, self.out_weights) * 64**-0.5
#         return inputs
#
#     def padding(self, padding):
#         padding = tf.expand_dims(padding, axis=-1)
#         return padding
#
#     def get_config(self):
#         return {"num_units": self.num_units, "dropout": self.dropout}


class Daedalus(tf.keras.Model):
    def __init__(self,
                 max_seq_len=30,
                 vocabulary_size=12000,
                 embedding_size=512,
                 batch_size=64,
                 num_units=512,
                 num_heads=6,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.4,
                 eos_id=1,
                 pad_id=0,
                 mask_id=2,
                 **kwargs):
        super(Daedalus, self).__init__(name="lip_reading")
        self.max_seq_len = max_seq_len
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.EOS_ID = eos_id
        self.PAD_ID = pad_id
        self.MASK_ID = mask_id

        self.embedding_block = hyper_layer.EmbeddingSharedWeights(
            vocab_size=self.vocabulary_size,
            num_units=self.num_units,
            pad_id=self.PAD_ID,
            name="word_embedding",
        )

        self.res_encoder = ResLinearEncoder(
            self.max_seq_len,
            self.vocabulary_size,
            self.embedding_size,
            self.batch_size,
            self.num_units,
            self.num_heads,
            self.num_encoder_layers,
            self.dropout,
            self.EOS_ID,
            self.PAD_ID,
        )
        # self.stacked_encoder.load_weights(
        #     tf.train.latest_checkpoint("/home/vivalavida/massive_data/model_checkpoint/model.10.ckpt"))
        self.stacked_encoder = LinearEncoder(
            self.max_seq_len,
            self.vocabulary_size,
            self.embedding_size,
            self.batch_size,
            self.num_units,
            self.num_heads,
            6,
            self.dropout,
            self.EOS_ID,
            self.PAD_ID,
        )
        # self.extra_encoder.load_weights(
        #     tf.train.latest_checkpoint("/home/vivalavida/massive_data/model_checkpoint/"))
        self.stacked_decoder = LinearDecoder(
            self.max_seq_len,
            self.vocabulary_size,
            self.embedding_size,
            self.batch_size,
            self.num_units,
            self.num_heads,
            self.num_decoder_layers,
            self.dropout,
            self.EOS_ID,
            self.PAD_ID,
        )

    def call(self, inputs, training, mode='LIP'):
        if len(inputs) == 2:
            img_input, tgt_input = inputs[0], inputs[1]
        else:
            img_input, tgt_input = inputs[0], None
        # batch_size = tf.shape(input=img_input)[0]
        with tf.name_scope("lip_reading"):
            Q = img_input
            if training:
                img_input_padding = tf.reshape(Q,
                                               [-1, PADDED_IMG, FEATURE_IMG])
            else:
                img_input_padding = tf.reshape(Q, [-1, 1, FEATURE_IMG])
            img_input_padding = tf.reduce_sum(img_input_padding, -1)
            attention_bias = hyper_util.get_padding_bias(img_input_padding)
            if mode == 'LIP':
                Q = self.res_encoder(
                    Q, attention_bias=attention_bias, training=training)
            # attention_bias = hyper_util.get_padding_bias(img_input)
            encoder_outputs = self.encode(
                Q, attention_bias, training, mode=mode)
            if tgt_input is None:
                return self.inference(encoder_outputs, attention_bias,
                                      training)
            else:
                logits = self.decode(tgt_input, encoder_outputs,
                                     attention_bias, training)
                return logits

    def encode(self, inputs, attention_bias, training, mode='LIP'):
        with tf.name_scope("encode"):
            ##############
            with tf.name_scope("add_pos_encoding"):
                if mode != 'LIP':
                    inputs = self.embedding_block(inputs)
                length = tf.shape(inputs)[1]
                pos_encoding = conf_fn.get_position_encoding(
                    length, self.num_units)
                encoder_inputs = inputs + pos_encoding
            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.dropout)
            # print(tf.reduce_sum(encoder_inputs, [1, 2]))
            encoder = self.stacked_encoder(
                encoder_inputs,
                attention_bias=attention_bias,
                training=training)
            return encoder

    def decode(self, targets, encoder_outputs, attention_bias, training):
        with tf.name_scope("decode"):
            decoder_inputs = targets
            decoder_inputs = self.embedding_block(decoder_inputs)
            with tf.name_scope("shift_targets"):
                # decoder_inputs = tf.pad(
                #     decoder_inputs, [[0, 0], [1, 0], [0, 0]],
                #     constant_values=0)[:, :-1, :]

                # for word training
                decoder_inputs = tf.zeros_like(decoder_inputs)
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += conf_fn.get_position_encoding(
                    length, self.num_units)
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.dropout)
            decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
                length)
            outputs = self.stacked_decoder(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training,
            )
            # for i, layer in enumerate(self.embeding_att):
            #     org = outputs
            #     in_project = layer[0]
            #     out_project = layer[1]
            #     norm = layer[2]
            #     outputs = in_project(norm(outputs))
            #     outputs = self.embedding_block.att_shared_weights(outputs)
            #     outputs = out_project(outputs)
            #     outputs = org + outputs
            #     # org = encoder_inputs
            # outputs = self.x_encode_norm(outputs)
            logits = self.embedding_block.linear(outputs)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        timing_signal = conf_fn.get_position_encoding(max_decode_length + 1,
                                                      self.num_units)
        decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = self.embedding_block(decoder_input)
            decoder_input += timing_signal[i:i + 1]
            self_attention_bias = decoder_self_attention_bias[:, :, i:i +
                                                              1, :i + 1]
            decoder_outputs = self.stacked_decoder(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache,
            )
            logits = self.embedding_block.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def inference(self, encoder_outputs, encoder_decoder_attention_bias,
                  training):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        # input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = self.max_seq_len

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.num_units]),
                "v": tf.zeros([batch_size, 0, self.num_units]),
            }
            for layer in range(self.num_decoder_layers)
        }
        cache["encoder_outputs"] = encoder_outputs
        cache[
            "encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary_size,
            beam_size=4,
            alpha=0.6,
            max_decode_length=max_decode_length,
            eos_id=self.EOS_ID,
        )
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

    def padding_bias(self, padding, eye=False):
        if eye:
            length = tf.shape(padding)[1]
            self_ignore = tf.eye(length, dtype=tf.float32)
            self_ignore = tf.expand_dims(
                tf.expand_dims(self_ignore, axis=0), axis=0)
            padding = tf.expand_dims(tf.expand_dims(padding, axis=1), axis=1)
            padding = tf.cast(
                tf.cast((self_ignore + padding), tf.bool), tf.float32)
        else:
            padding = tf.expand_dims(tf.expand_dims(padding, axis=1), axis=1)

        attention_bias = padding * (-1e9)
        return attention_bias

    def get_config(self):
        c = {
            "max_seq_len": self.max_seq_len,
            "vocabulary_size": self.vocabulary_size,
            "embedding_size": self.embedding_size,
            "batch_size": self.batch_size,
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_decoder_layers": self.num_decoder_layers,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c

    def randomly_pertunate_input(self, x):
        determinater = np.random.randint(10)
        if determinater > 3:
            return x
        else:
            index = np.random.randint(2, size=(1, 80))
            x = x * index
        return x

    def padding(self, padding):
        padding = tf.expand_dims(padding, axis=-1)
        return padding

    def window_bias(self, length):
        with tf.name_scope("decoder_self_attention_bias"):
            valid_locs = tf.linalg.band_part(tf.ones([length, length]), 5, 5)
            valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
            decoder_bias = -1e9 * (1.0 - valid_locs)
        return decoder_bias
