# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer

# from tensorflow.python.layers import core as core_layer
# import numpy as np
# from hyper_and_conf import hyper_beam_search as beam_search


class Transformer_Encoder(tf.keras.layers.Layer):
    def __init__(self,
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
                 fusion=False):
        super(Transformer_Encoder, self).__init__(name='transformer_encoder')
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

        self.position_encoding = hyper_layer.Positional_Encoding(
            2 * self.num_units)
        self.en_att = []
        self.en_ffn = []
        self.fusion_kernel = []
        self.fusion_kernel_2nd = []
        self.fusion_ffn = []
        self.fusion_bias = []
        self.fusion_bias_2nd = []
        self.negtive_infinit = -1e32
        self.norm = hyper_layer.LayerNorm()
        self.normx2 = hyper_layer.LayerNorm()
        for i in range(self.num_encoder_layers):
            self.en_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="enc_multi_att_%d" % i))
            self.en_ffn.append(
                hyper_layer.Feed_Forward_Network(
                    num_units=4 * self.num_units,
                    dropout=self.dropout,
                    name="enc_ffn_%d" % i))
            if fusion:
                self.fusion_bias.append(
                    self.add_variable(
                        shape=self.num_units,
                        name="fusion_bias_%d" % i,
                        initializer=tf.keras.initializers.get('zeros')))
                self.fusion_kernel.append(
                    self.add_variable(
                        name='fusion_block_%d' % i,
                        shape=[2 * self.num_units, self.num_units],
                        initializer=tf.keras.initializers.get(
                            "glorot_uniform")))
                self.fusion_ffn.append(
                    hyper_layer.Feed_Forward_Network(
                        num_units=4 * self.num_units,
                        dropout=self.dropout,
                        name="fusion_ffn_%d" % i))

    def call(self,
             inputs,
             padding_matrix=None,
             fusion_matrix=None,
             length=None,
             position=True,
             training=False):
        # inputs= inputs
        # inputs = Q
        with tf.compat.v1.name_scope("encoder"):
            self.fusion_res = []
            if training is False:
                dropout_mask_inputs = tf.keras.backend.dropout(
                    tf.ones_like(inputs), self.dropout)
                inputs = inputs * dropout_mask_inputs

            if length is None:
                length = tf.shape(input=inputs)[1]
            if padding_matrix is not None:
                padding_mask_bias = self.padding_bias(padding_matrix)
            else:
                padding_mask_bias = 0
            outputs = inputs
            for i in range(self.num_encoder_layers):
                with tf.compat.v1.name_scope('layer_%d' % i):
                    # block 1
                    if fusion_matrix is not None:
                        with tf.name_scope('fusion_block_%d' % i):
                            org = outputs
                            outputs = tf.keras.layers.concatenate(
                                [outputs, fusion_matrix], axis=-1)
                            if i == 0:
                                if position:
                                    positional_input = self.position_encoding(
                                        length)
                                    outputs = outputs + positional_input
                                    if training is not False:
                                        dropout_mask_inputs = tf.keras.backend.dropout(
                                            tf.ones_like(outputs),
                                            self.dropout)
                                        outputs = outputs * dropout_mask_inputs
                            norm = self.normx2(outputs)
                            res = self.fusion_ffn[i](norm)
                            res = tf.keras.layers.Activation("relu")(res)
                            res = tf.keras.backend.dot(res,
                                                       self.fusion_kernel[i])
                            res = tf.keras.layers.Activation("relu")(res)
                            res = res + self.fusion_bias[i]

                            self.fusion_res.append(res)

                            if training is not False:
                                dropout_mask_inputs = tf.keras.backend.dropout(
                                    tf.ones_like(res), self.dropout)
                                res = res * dropout_mask_inputs
                            outputs = res + org
                    # # block 2
                    # norm = self.norm(outputs)
                    # multi_att = self.en_att[i](
                    #     norm,
                    #     K_V=(norm, norm),
                    #     bias=padding_mask_bias,
                    #     training=training)
                    # if training is not False:
                    #     dropout_mask_inputs = tf.keras.backend.dropout(
                    #         tf.ones_like(multi_att), self.dropout)
                    #     multi_att = multi_att * dropout_mask_inputs
                    # res = outputs + multi_att
                    # # block 3
                    # multi_att = self.norm(res)
                    # multi_att = self.en_ffn[i](multi_att, training=training)
                    # if training is not False:
                    #     dropout_mask_inputs = tf.keras.backend.dropout(
                    #         tf.ones_like(multi_att), self.dropout)
                    #     multi_att = multi_att * dropout_mask_inputs
                    # outputs = res + multi_att
        return self.norm(outputs)

    def padding_bias(self, padding):
        attention_bias = padding * self.negtive_infinit
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias

    def get_config(self):
        config = super(Transformer_Encoder, self).get_config()
        c = {
            'max_seq_len': self.max_seq_len,
            'vocabulary_size': self.vocabulary_size,
            'embedding_size': self.embedding_size,
            'batch_size': self.batch_size,
            'num_units': self.num_units,
            'num_heads': self.num_heads,
            'num_encoder_layers': self.num_encoder_layers,
            'dropout': self.dropout
        }
        config.update(c)
        return config

    def get_fusion(self):
        return self.fusion_res


class Transformer_Decoder(tf.keras.layers.Layer):
    def __init__(self,
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
                 shared_embedding=None):
        super(Transformer_Decoder, self).__init__(name='transformer_decoder')
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

        self.position_encoding = hyper_layer.Positional_Encoding(
            self.num_units)
        self.de_att = []
        self.de_ffn = []
        self.de_mask_att = []
        self.negtive_infinit = -1e32
        self.norm = hyper_layer.LayerNorm()
        for i in range(self.num_decoder_layers):
            self.de_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=False,
                    name="de_multi_att_%d" % i))
            self.de_ffn.append(
                hyper_layer.Feed_Forward_Network(
                    num_units=4 * self.num_units,
                    dropout=self.dropout,
                    name="dec_ffn_%d" % i))
            self.de_mask_att.append(
                hyper_layer.Multi_Head_Attention(
                    num_heads=self.num_heads,
                    num_units=self.num_units,
                    dropout=self.dropout,
                    masked_attention=True,
                    name="masked_multi_att_%d" % i))

    def call(self,
             inputs,
             enc=None,
             self_mask_bias=None,
             padding_matrix=None,
             length=None,
             cache=None,
             training=False):
        with tf.compat.v1.name_scope('decoder'):
            if enc is None:
                assert ('Using maksed_attention, please give enc')
            if length is None:
                length = tf.shape(input=inputs)[1]
            # src_input = tf.multiply(tf.cast(inputs, tf.float32), self.num_units**0.5)
            src_input = inputs * self.num_units**0.5
            if self_mask_bias is None:
                self_mask_bias = self.masked_self_attention_bias(length)
            if padding_matrix is not None:
                padding_mask_bias = self.padding_bias(padding_matrix)
            else:
                padding_mask_bias = 0
            positional_input = self.position_encoding(length)

            inputs = src_input + positional_input
            # outputs = self.norm(inputs)
            if training is not False:
                dropout_mask_inputs = tf.keras.backend.dropout(
                    tf.ones_like(inputs), self.dropout)
                inputs = inputs * dropout_mask_inputs
                # dropout_mask_enc = tf.keras.backend.dropout(
                #     tf.ones_like(enc), self.dropout)
                # enc = enc * dropout_mask_enc
            K_V = self.norm(inputs)
            outputs = inputs
            # K_V = inputs
            i = 0

            if cache is not None:
                # Combine cached keys and values with new keys and values.
                K_V = tf.concat((cache[str(0)], outputs), axis=1)
                # Update cache
                for i in range(self.num_decoder_layers):
                    cache[str(i)] = K_V
            for i in range(self.num_decoder_layers):
                with tf.compat.v1.name_scope('layer_%d' % i):
                    # block 1
                    norm = self.norm(outputs)
                    de_outputs = self.de_mask_att[i](
                        norm,
                        K_V=(K_V, K_V),
                        bias=self_mask_bias,
                        cache=cache,
                        training=training)
                    if training is not False:
                        dropout_mask_inputs = tf.keras.backend.dropout(
                            tf.ones_like(de_outputs), self.dropout)
                        de_outputs = de_outputs * dropout_mask_inputs
                    res = outputs + de_outputs
                    # block 2
                    de_outputs = self.norm(res)
                    multi_att = self.de_att[i](
                        de_outputs,
                        K_V=(enc, enc),
                        bias=padding_mask_bias,
                        training=training)
                    if training is not False:
                        dropout_mask_inputs = tf.keras.backend.dropout(
                            tf.ones_like(multi_att), self.dropout)
                        multi_att = multi_att * dropout_mask_inputs
                    res = res + multi_att
                    # block 3
                    multi_att = self.norm(res)
                    multi_att = self.de_ffn[i](multi_att, training=training)
                    if training is not False:
                        dropout_mask_inputs = tf.keras.backend.dropout(
                            tf.ones_like(multi_att), self.dropout)
                        multi_att = multi_att * dropout_mask_inputs
                    outputs = res + multi_att

                    # outputs = self.norm(outputs + multi_att)
            return self.norm(outputs)

    def get_config(self):
        config = super(Transformer_Decoder, self).get_config()
        c = {
            'max_seq_len': self.max_seq_len,
            'vocabulary_size': self.vocabulary_size,
            'embedding_size': self.embedding_size,
            'batch_size': self.batch_size,
            'num_units': self.num_units,
            'num_heads': self.num_heads,
            'num_decoder_layers': self.num_decoder_layers,
            'dropout': self.dropout
        }
        config.update(c)
        return config

    def masked_self_attention_bias(self, length):
        valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = self.negtive_infinit * (1.0 - valid_locs)
        return decoder_bias

    def padding_bias(self, padding):
        attention_bias = padding * self.negtive_infinit
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
        return attention_bias

    def symbols_to_logits_fn(self, max_seq_len, shared_embedding):
        inference_position = self.position_encoding(max_seq_len)
        masked_attention_bias = self.masked_self_attention_bias(max_seq_len)

        def body(ids, i, cache):
            decoder_input = ids[:, -1:]
            decoder_input = shared_embedding(decoder_input)
            self_mask_bias = masked_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_input += inference_position[i:i + 1]
            if self.embedding_size != self.num_units:
                decoder_input = self.src_dense(decoder_input)
            # Preprocess decoder input by getting embeddings and adding timing signal.
            outputs = self.call(
                decoder_input,
                cache['enc'],
                padding_matrix=cache['src_padding'],
                self_mask_bias=self_mask_bias,
                cache=cache,
                training=False)
            # projection = self.projection(outputs)
            logits = shared_embedding.linear(outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return body


# class Prometheus(tf.keras.Model):
#     """
#         Transformer
#     """
#
#     def __init__(self,
#                  max_seq_len,
#                  vocabulary_size,
#                  embedding_size=512,
#                  batch_size=64,
#                  num_units=512,
#                  num_heads=6,
#                  num_encoder_layers=6,
#                  num_decoder_layers=6,
#                  dropout=0.4,
#                  eos_id=1,
#                  pad_id=0,
#                  external_embedding=False,
#                  shared_embedding=None):
#         super(Prometheus, self).__init__(name='transformer')
#         self.max_seq_len = max_seq_len
#         self.vocabulary_size = vocabulary_size
#         # self.vocabulary_size = 32000
#         # self.src_vocabulary_size = src_vocabulary_size
#         # self.tgt_vocabulary_size = tgt_vocabulary_size
#         self.embedding_size = embedding_size
#         self.batch_size = batch_size
#         self.num_units = num_units
#         self.num_heads = num_heads
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.dropout = dropout
#         self.eos_id = eos_id
#         self.pad_id = pad_id
#
#         self.Eecoder = Transformer_Encoder(
#             max_seq_len=max_seq_len,
#             vocabulary_size=vocabulary_size,
#             embedding_size=embedding_size,
#             batch_size=batch_size,
#             num_units=num_units,
#             num_heads=num_heads,
#             num_encoder_layers=num_encoder_layers,
#             dropout=dropout,
#             eos_id=eos_id,
#             pad_id=pad_id,
#         )
#         self.Decoder = Transformer_Decoder(
#             max_seq_len=max_seq_len,
#             vocabulary_size=vocabulary_size,
#             embedding_size=embedding_size,
#             batch_size=batch_size,
#             num_units=num_units,
#             num_heads=num_heads,
#             num_decoder_layers=num_decoder_layers,
#             dropout=dropout,
#             eos_id=eos_id,
#             pad_id=pad_id,
#             shared_embedding=shared_embedding
#         )
#         self.shared_embedding = shared_embedding
#         # self.shared_embedding = hyper_layer.EmbeddingSharedWeights(
#         #     self.vocabulary_size, self.embedding_size, self.pad_id)
#         if self.embedding_size != self.num_units:
#             self.src_dense = tf.keras.layers.Dense(
#                 self.num_units, name='src_embedding_dense')
#             self.tgt_dense = tf.keras.layers.Dense(
#                 self.embedding_size, name='tgt_embedding_dense')
#
#     def build(self, input_shape):
#         self.build = True
#
#     def call(self, inputs, training=False):
#         # src, tgt = tf.split(inputs, 2, 0)
#         # inputs = (src, tgt)
#         if training:
#             return self.train_model(inputs)
#         else:
#             return self.inference_model(inputs, training=False)
#
#     def train_model(self, inputs, training=True):
#         # src_input, tgt = tf.split(inputs, 2)
#         src_input, tgt = inputs
#         src_input = tf.cast(src_input, tf.int64)
#         tgt = tf.cast(tgt, tf.int64)
#         src_padding = tf.cast(
#             tf.equal(src_input, self.pad_id), dtype=tf.float32)
#         embedding_src_input = self.shared_embedding(src_input)
#         embedding_tgt_input = self.shared_embedding(tgt)
#         embedding_tgt_input = tf.pad(
#             tensor=embedding_tgt_input, paddings=[[0, 0], [1, 0],
#                                                   [0, 0]])[:, :-1, :]
#         if self.embedding_size != self.num_units:
#             embedding_src_input = self.src_dense(embedding_src_input)
#             embedding_tgt_input = self.tgt_dense(embedding_tgt_input)
#
#         enc = self.Encoder(
#             embedding_src_input, padding_matrix=src_padding, training=training)
#         dec = self.Decoder(
#             embedding_tgt_input,
#             enc,
#             padding_matrix=src_padding,
#             training=training)
#         # projection = self.projection(self.outputs)
#         # logits = tf.keras.layers.Softmax()(projection)
#         logits = self.shared_embedding.linear(dec)
#         return logits
#
#     def inference_model(self, inputs, training):
#         if isinstance(inputs, list) or isinstance(inputs, tuple):
#             src_input, _ = inputs
#         else:
#             src_input = inputs
#
#         initial_size = tf.shape(input=inputs)[0]
#         src_padding = tf.cast(
#             tf.equal(src_input, self.pad_id), dtype=tf.float32)
#         src_input = tf.cast(src_input, tf.int32)
#         embedding_src_input = self.shared_embedding(src_input)
#         if self.embedding_size != self.num_units:
#             embedding_src_input = self.src_dense(embedding_src_input)
#         enc = self.Encoder(
#             embedding_src_input, padding_matrix=src_padding, training=False)
#         # initial_ids = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.int32)
#         initial_ids = tf.zeros([initial_size], dtype=tf.int32)
#
#         cache = dict()
#         cache['enc'] = enc
#         cache['src_padding'] = src_padding
#         for i in range(self.num_decoder_layers):
#             cache[str(i)] = tf.zeros([initial_size, 0, self.num_units])
#             # cache[str(i)] = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.float32)
#         # cache['K'] = tf.zeros([self.batch_size, 0, self.num_units])
#         # cache['V'] = tf.zeros([self.batch_size, 0, self.num_units])
#         logits_body = self.Decoder.symbols_to_logits_fn(self.max_seq_len)
#         decoded_ids, scores = beam_search.sequence_beam_search(
#             symbols_to_logits_fn=logits_body,
#             initial_ids=initial_ids,
#             initial_cache=cache,
#             vocab_size=self.vocabulary_size,
#             beam_size=4,
#             alpha=0.6,
#             max_decode_length=self.max_seq_len,
#             eos_id=self.eos_id)
#         # Get the top sequence for each batch element
#         top_decoded_ids = decoded_ids[:, 0, 1:]
#
#         # top_scores = scores[:, 0]
#         # self.attention = cache['attention']
#         return top_decoded_ids
