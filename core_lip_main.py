# encoding=utf8
import tensorflow as tf
import core_Transformer_model
from hyper_and_conf import hyper_layer
from hyper_and_conf import hyper_beam_search as beam_search
# from hyper_and_conf import hyper_param
from core_FusionBlock_model import Fusion_Block

# tf.enable_eager_execution()


class Daedalus(tf.keras.layers.Layer):
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
        super(Daedalus, self).__init__(name='lip_reading')
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

    def get_encoder(
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
        encoder = core_Transformer_model.Transformer_Encoder(
            max_seq_len=max_seq_len,
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            batch_size=batch_size,
            num_units=num_units,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            eos_id=eos_id,
            pad_id=pad_id,
            fusion=True)
        return encoder

    def get_decoder(
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
        decoder = core_Transformer_model.Transformer_Decoder(
            max_seq_len=max_seq_len,
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            batch_size=batch_size,
            num_units=num_units,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            eos_id=eos_id,
            pad_id=pad_id,
            shared_embedding=self.word_embedding)
        return decoder

    def build(self, input_shape):
        self.word_embedding = hyper_layer.EmbeddingSharedWeights(
            vocab_size=self.vocabulary_size,
            hidden_size=self.num_units,
            pad_id=self.PAD_ID,
            name='word_embedding')
        # self.pre_encoder = self.get_encoder(self)
        # self.pre2_encoder = self.get_encoder(self)
        self.fusion_block = Fusion_Block(self.num_units, self.dropout)
        self.Encoder = self.get_encoder(
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
        self.Decoder = self.get_decoder(
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

        self.kernel_initializer = tf.keras.initializers.get("glorot_uniform")
        self.Q = self.add_variable(
            name='Q',
            shape=[4096, 4 * self.num_units],
            initializer=self.kernel_initializer)
        self.Q_2nd = self.add_variable(
            name='Q_2nd',
            shape=[4 * self.num_units, self.num_units],
            initializer=self.kernel_initializer)
        self.norm = hyper_layer.LayerNorm()
        self.build = True

    def call(self, inputs, training=True):
        if training is False:
            if isinstance(inputs, tuple) or isinstance(inputs, list):
                img_input, tgt_input = inputs
            else:
                img_input = inputs
            word_id = self.inference_model(img_input, training)
            return word_id
        else:
            logits = self.train_model(inputs, training=True)
            return logits

    def train_model(self, inputs, training=True):
        img_input, tgt_input = inputs
        img_input = img_input * self.num_units**0.5
        img_input_padding = tf.cast(
            tf.equal(img_input, self.PAD_ID), dtype=tf.float32)[:, :, 0]
        dropout_mask = tf.keras.backend.dropout(
            tf.ones_like(img_input), self.dropout)
        img_input = img_input * dropout_mask
        Q = tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(
            tf.keras.backend.dot(img_input, self.Q)))
        dropout_mask = tf.keras.backend.dropout(tf.ones_like(Q), self.dropout)
        Q = Q * dropout_mask
        Q = tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(
            tf.keras.backend.dot(Q, self.Q_2nd)))
        dropout_mask = tf.keras.backend.dropout(tf.ones_like(Q), self.dropout)
        Q = Q * dropout_mask
        mask_id = self.MASK_ID
        mask_words = tf.zeros_like(Q[:, :, 0], dtype=tf.int32) + mask_id
        mask_embedding = self.word_embedding(mask_words)
        Q = self.fusion_block((mask_embedding, Q), training=True)
        # mask_embedding = mask_embedding * self.num_units**0.5

        encoder_out = self.Encoder(
            Q,
            img_input_padding,
            # fusion_matrix=mask_embedding,
            position=False,
            training=True)

        embedding_tgt_input = self.word_embedding(tgt_input)
        embedding_tgt_input = tf.pad(
            tensor=embedding_tgt_input, paddings=[[0, 0], [1, 0],
                                                  [0, 0]])[:, :-1, :]
        decoder_out = self.Decoder(
            embedding_tgt_input,
            encoder_out,
            padding_matrix=img_input_padding,
            training=True)
        logits = self.word_embedding.linear(decoder_out)
        # model = tf.keras.Model([img_input, tgt_input], logits)
        return logits

    def inference_model(self, img_input, training=False):
        img_input_padding = tf.cast(
            tf.equal(img_input, self.PAD_ID), dtype=tf.float32)[:, :, 0]

        dropout_mask = tf.keras.backend.dropout(
            tf.ones_like(img_input), self.dropout)
        img_input = img_input * dropout_mask

        Q = tf.keras.activations.relu(tf.keras.backend.dot(img_input, self.Q))
        dropout_mask = tf.keras.backend.dropout(tf.ones_like(Q), self.dropout)
        Q = Q * dropout_mask
        Q = tf.keras.activations.relu(tf.keras.backend.dot(Q, self.Q_2nd))
        dropout_mask = tf.keras.backend.dropout(tf.ones_like(Q), self.dropout)
        Q = Q * dropout_mask

        img_input_padding = tf.cast(
            tf.equal(Q, self.PAD_ID), dtype=tf.float32)[:, :, 0]
        initial_size = tf.shape(input=Q)[0]

        mask_id = self.MASK_ID
        mask_words = tf.zeros_like(Q[:, :, 0], dtype=tf.int32) + mask_id
        mask_embedding = self.word_embedding(mask_words)
        Q = self.fusion_block((mask_embedding, Q), training=True)

        enc = self.Encoder(
            Q, img_input_padding, position=False, training=False)
        # initial_ids = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.int32)
        initial_ids = tf.zeros([initial_size], dtype=tf.int32)

        cache = dict()
        cache['enc'] = enc
        cache['src_padding'] = img_input_padding
        for i in range(self.num_decoder_layers):
            cache[str(i)] = tf.zeros([initial_size, 0, self.num_units])
            # cache[str(i)] = tf.constant( self.sos_id, shape=[self.batch_size], dtype=tf.float32)
        logits_body = self.Decoder.symbols_to_logits_fn(
            self.max_seq_len, self.word_embedding)
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=logits_body,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.vocabulary_size,
            beam_size=4,
            alpha=0.6,
            max_decode_length=self.max_seq_len,
            eos_id=self.EOS_ID)
        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        print(scores)
        print(decoded_ids)

        # top_scores = scores[:, 0]
        # self.attention = cache['attention']
        return top_decoded_ids

    def get_config(self):
        config = super(Daedalus, self).get_config()
        c = {
            'max_seq_len': self.max_seq_len,
            'vocabulary_size': self.vocabulary_size,
            'embedding_size': self.embedding_size,
            'batch_size': self.batch_size,
            'num_units': self.num_units,
            'num_heads': self.num_heads,
            'num_decoder_layers': self.num_decoder_layers,
            'num_encoder_layers': self.num_encoder_layers,
            'dropout': self.dropout
        }
        config.update(c)
        return config


# if '__name__' == '__main__':
#     from hyper_and_conf import hyper_param
#     import tfrecoder_generator
#     import core_model_initializer
#     hp = hyper_param.HyperParam(mode='test')
#     model = main_model(hp)

#     vgg16 = get_vgg()  # step = tf.shape(vgg16_input)
#     # step = vgg16_input.get_shape().as_list()[1]
#     output = []
#     vgg16_flatten = vgg16.get_layer('flatten')
#     vgg16_output = vgg16_flatten.output
#     vgg16.input
#     model = tf.keras.Model(vgg16.input, vgg16_output)
#     test = tf.constant(0.0, shape=[2, 224, 224, 3])
#     model(test)
