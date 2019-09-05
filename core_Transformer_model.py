# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer
from core_resnet import identity_block, conv_block
from hyper_and_conf import hyper_util

# from tensorflow.python.layers import core as core_layer
# import numpy as np
# from hyper_and_conf import hyper_beam_search as beam_search
PADDED_IMG = 50
PADDED_TEXT = 1


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
                dropout=self.dropout)
            ffn = hyper_layer.Feed_Forward_Network(
                num_units=4 * self.num_units, dropout=self.dropout)
            # ffn = hyper_layer.CNN_FNN(self.num_units, self.dropout)

            self.stacked_encoder.append([
                hyper_layer.NormBlock(self_attention, self.dropout),
                hyper_layer.NormBlock(ffn, self.dropout),
                # ffn
            ])
        self.encoder_output = hyper_layer.LayerNorm()
        super(LinearEncoder, self).build(input_shape)

    def call(self, inputs, attention_bias, training):
        # inputs= inputs
        # inputs = Q
        length = tf.shape(inputs)[1]
        with tf.name_scope("stacked_encoder"):
            for index, layer in enumerate(self.stacked_encoder):
                with tf.name_scope("layer_%d" % index):
                    self_att = layer[0]
                    ffn = layer[1]
                    inputs = self_att(
                        inputs, attention_bias, training=training)
                    # inputs = tf.reshape(inputs, [-1, 1, 1, self.num_units])
                    inputs = ffn(inputs, training=training)
                    # inputs = tf.reshape(inputs, [-1, length, self.num_units])
                    # inputs = ffn(inputs, training=training)
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
                dropout=self.dropout)
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


class Transformer(tf.keras.Model):
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
        super(Transformer, self).__init__(name='Transformer')
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
        self.embedding_softmax_layer = hyper_layer.EmbeddingSharedWeights(
            self.vocabulary_size, self.num_units, pad_id=self.PAD_ID)
        self.encoder_stack = LinearEncoder(
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
        self.decoder_stack = LinearDecoder(
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

    def call(self, inputs, training):
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = hyper_util.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias, training)
            # Generate output sequence if targets is None, or return logits if target
            # sequence is known.
            if targets is None:
                return self.predict(encoder_outputs, attention_bias, training)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias,
                                     training)
                return logits

    def encode(self, inputs, attention_bias, training):
        """Generate continuous representation for inputs.
    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.
    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_softmax_layer(inputs)
            # inputs_padding = hyper_util.get_padding(inputs)
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = hyper_util.get_position_encoding(
                    length, self.num_units)
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.dropout)

            return self.encoder_stack(
                encoder_inputs, attention_bias, training=training)

    def decode(self, targets, encoder_outputs, attention_bias, training):
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs,
                                        [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += hyper_util.get_position_encoding(
                    length, self.num_units)
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.dropout)

            # Run values
            decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
                length)
            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                decoder_self_attention_bias,
                attention_bias,
                training=training)
            logits = self.embedding_softmax_layer.linear(outputs)
            # logits = tf.cast(logits, tf.float32)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = hyper_util.get_position_encoding(
            max_decode_length + 1, self.num_units)
        decoder_self_attention_bias = hyper_util.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.
      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.
      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i +
                                                              1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input,
                cache.get("encoder_outputs"),
                self_attention_bias,
                cache.get("encoder_decoder_attention_bias"),
                training=training,
                cache=cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias,
                training):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        max_decode_length = self.max_seq_len

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length, training)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.num_units]),
                "v": tf.zeros([batch_size, 0, self.num_units])
            }
            for layer in range(self.num_decoder_layers)
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache[
            "encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = hyper_beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary_size,
            beam_size=4,
            alpha=0.6,
            max_decode_length=max_decode_length,
            eos_id=self.EOS_ID)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}
