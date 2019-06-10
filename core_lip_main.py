# encoding=utf8
import tensorflow as tf
from core_Transformer_model import LinearEncoder, LinearDecoder
from hyper_and_conf import hyper_layer
from hyper_and_conf import hyper_beam_search as beam_search

# from hyper_and_conf import hyper_param
from core_FusionBlock_model import Fusion_Block
from hyper_and_conf import hyper_util, conf_fn


# tf.enable_eager_execution()
class VGG_POST(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout):
        self.num_units = num_units
        self.dropout = dropout
        super(VGG_POST, self).__init__()

    def build(self, input_shape):
        self.vgg_dense_1 = tf.keras.layers.Dense(
            4096, use_bias=True, activation=tf.nn.relu, name="vgg_dense_1")
        self.vgg_dense_2 = tf.keras.layers.Dense(
            self.num_units, use_bias=True, name="vgg_dense_2")
        self.vgg_dense_1_post_norm = hyper_layer.LayerNorm()
        # self.vgg_dense_2_post_norm = hyper_layer.LayerNorm()
        # self.inputs_projection = tf.keras.layers.Conv1D(
        #     self.num_units,
        #     1,
        #     use_bias=False,
        #     kernel_initializer='he_normal',
        # )
        self.inputs_projection = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="projection")
        self.res_projection = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="res_projection")
        self.output_norm = hyper_layer.LayerNorm()
        super(VGG_POST, self).build(input_shape)

    def call(self, inputs, training):
        inputs = tf.transpose(inputs, [1, 0, 2])
        org = self.inputs_projection(inputs)
        # org = inputs
        inputs = self.vgg_dense_1(inputs)
        if training:
            inputs = tf.nn.dropout(inputs, rate=self.dropout)
        inputs = self.res_projection(self.vgg_dense_2(inputs))
        # inputs = self.vgg_dense_2_post_norm(inputs)
        if training:
            inputs = tf.nn.dropout(inputs, rate=self.dropout)
        inputs = self.output_norm(org + inputs)
        return tf.transpose(inputs, [1, 0, 2])

    def get_config(self):
        return {"num_units": self.num_units, "dropout": self.dropout}


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
        self.fusion_block = Fusion_Block(self.num_units, self.dropout)
        self.stacked_encoder = LinearEncoder(
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
        # self.temp = self.get_encoder(self)
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
        self.norm = hyper_layer.LayerNorm()
        self.vgg_post = VGG_POST(self.num_units, self.dropout)

    # def build(self, input_shape):

    def call(self, inputs, training):
        if len(inputs) == 2:
            img_input, tgt_input = inputs[0], inputs[1]
        else:
            img_input, tgt_input = inputs[0], None

        with tf.name_scope("lip_reading"):
            img_input_padding = tf.cast(
                tf.equal(tf.reduce_sum(img_input, -1), 0.0), dtype=tf.float32)
            Q = self.vgg_post(img_input, training=training)
            mask_id = self.MASK_ID

            mask_words = tf.cast((1 - img_input_padding) * mask_id, tf.int32)
            mask_embedding = self.embedding_block(mask_words)
            Q = self.fusion_block((Q, mask_embedding), training=training)
            attention_bias = self.padding_bias(img_input_padding)

            # Q = img_input
            encoder_outputs = self.encode(Q, attention_bias, training)
            if tgt_input is None:
                return self.inference(encoder_outputs, attention_bias,
                                      training)
            else:
                logits = self.decode(tgt_input, encoder_outputs,
                                     attention_bias, training)
                return logits

    def encode(self, inputs, attention_bias, training):
        with tf.name_scope("encode"):
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(inputs)[1]
                pos_encoding = conf_fn.get_position_encoding(
                    length, self.num_units)
                encoder_inputs = inputs + pos_encoding
            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.dropout)

            return self.stacked_encoder(
                encoder_inputs, attention_bias, training=training)

    def decode(self, targets, encoder_outputs, attention_bias, training):
        with tf.name_scope("decode"):
            decoder_inputs = self.embedding_block(targets)
            with tf.name_scope("shift_targets"):
                decoder_inputs = tf.pad(decoder_inputs,
                                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
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
            for layer in range(self.num_units)
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

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}

    def padding_bias(self, padding):
        attention_bias = padding * (-1e9)
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
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
