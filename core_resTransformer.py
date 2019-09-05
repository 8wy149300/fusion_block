# encoder=utf8
import tensorflow as tf
from hyper_and_conf import hyper_layer, hyper_util, conf_fn
# from core_resnet import identity_block, conv_block
# from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

# from tensorflow.python.layers import core as core_layer
# import numpy as np
# from hyper_and_conf import hyper_beam_search as beam_search
# PADDED_IMG = 150
# PADDED_TEXT = 80
PADDED_IMG = 50
PADDED_TEXT = 1


class ResLinearEncoder(tf.keras.layers.Layer):
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
        super(ResLinearEncoder, self).__init__(name="linear_encoder")
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
        self.avg_self_attention_64 = hyper_layer.NormBlock(
            hyper_layer.SelfAttention(
                num_heads=self.num_heads, num_units=16, dropout=self.dropout),
            self.dropout)
        self.avg_self_attention_256 = hyper_layer.NormBlock(
            hyper_layer.SelfAttention(
                num_heads=self.num_heads, num_units=64, dropout=self.dropout),
            self.dropout)
        self.avg_self_attention_512 = hyper_layer.NormBlock(
            hyper_layer.SelfAttention(
                num_heads=self.num_heads, num_units=128, dropout=self.dropout),
            self.dropout)
        self.avg_self_attention_1024 = hyper_layer.NormBlock(
            hyper_layer.SelfAttention(
                num_heads=self.num_heads, num_units=256, dropout=self.dropout),
            self.dropout)
        self.avg_self_attention_2048 = hyper_layer.NormBlock(
            hyper_layer.SelfAttention(
                num_heads=self.num_heads, num_units=512, dropout=self.dropout),
            self.dropout)

        self.self_attention_64 = hyper_layer.NormBlock(
            hyper_layer.Pixel_Self_Att(
                num_heads=self.num_heads, num_units=16, dropout=self.dropout),
            self.dropout)
        self.self_attention_256 = hyper_layer.NormBlock(
            hyper_layer.Pixel_Self_Att(
                num_heads=self.num_heads, num_units=64, dropout=self.dropout),
            self.dropout)
        self.self_attention_512 = hyper_layer.NormBlock(
            hyper_layer.Pixel_Self_Att(
                num_heads=self.num_heads, num_units=128, dropout=self.dropout),
            self.dropout)
        self.self_attention_1024 = hyper_layer.NormBlock(
            hyper_layer.Pixel_Self_Att(
                num_heads=self.num_heads, num_units=256, dropout=self.dropout),
            self.dropout)
        self.self_attention_2048 = hyper_layer.NormBlock(
            hyper_layer.Pixel_Self_Att(
                num_heads=self.num_heads, num_units=512, dropout=self.dropout),
            self.dropout)
        # ffn = hyper_layer.Feed_Forward_Network(
        #     num_units=4 * self.num_units, dropout=self.dropout)
        self.norm_init = hyper_layer.LayerNorm()
        self.norm_64 = hyper_layer.LayerNorm()
        self.norm_256 = hyper_layer.LayerNorm()
        self.norm_512 = hyper_layer.LayerNorm()
        self.norm_1024 = hyper_layer.LayerNorm()
        self.norm_2048 = hyper_layer.LayerNorm()

        # self.project_64_pixel = hyper_layer.CNN_FNN(16, self.dropout, kernel=1)
        # self.project_256_pixel = hyper_layer.CNN_FNN(
        #     64, self.dropout, kernel=2)
        # self.project_512_pixel = hyper_layer.CNN_FNN(128, self.dropout, kernel=1)
        # self.project_1024_pixel = hyper_layer.CNN_FNN(256, self.dropout, kernel=1)
        # self.project_2048_pixel = hyper_layer.CNN_FNN(512, self.dropout, kernel=1)
        #
        # self.project_64_avg = hyper_layer.CNN_FNN(16, self.dropout, kernel=1)
        # self.project_256_avg = hyper_layer.CNN_FNN(64, self.dropout, kernel=1)
        # self.project_512_avg = hyper_layer.CNN_FNN(128, self.dropout, kernel=1)
        # self.project_1024_avg = hyper_layer.CNN_FNN(
        #     256, self.dropout, kernel=1)
        # self.project_2048_avg = hyper_layer.CNN_FNN(
        #     512, self.dropout, kernel=1)
        self.project_64_pixel = tf.keras.layers.Dense(16, activation='relu')
        self.project_256_pixel = tf.keras.layers.Dense(64, activation='relu')
        self.project_512_pixel = tf.keras.layers.Dense(128, activation='relu')
        self.project_1024_pixel = tf.keras.layers.Dense(256, activation='relu')
        self.project_2048_pixel = tf.keras.layers.Dense(512, activation='relu')

        self.project_64_avg = tf.keras.layers.Dense(16, activation='relu')
        self.project_256_avg = tf.keras.layers.Dense(64, activation='relu')
        self.project_512_avg = tf.keras.layers.Dense(128, activation='relu')
        self.project_1024_avg = tf.keras.layers.Dense(256, activation='relu')
        self.project_2048_avg = tf.keras.layers.Dense(512, activation='relu')

        # self.project_64 = tf.keras.layers.Dense(16, activation='relu')
        # self.project_256 = tf.keras.layers.Dense(64, activation='relu')
        # self.project_512 = tf.keras.layers.Dense(128, activation='relu')
        # self.project_1024 = tf.keras.layers.Dense(256, activation='relu')
        # self.project_2048 = tf.keras.layers.Dense(512, activation='relu')
        # self.project_final = hyper_layer.CNN_FNN(
        #     self.num_units, self.dropout, kernel=1)
        self.project_final = tf.keras.layers.Dense(self.num_units)
        self.project_64 = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(16 * 8, self.dropout),
            self.dropout)

        self.project_256 = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(64 * 8, self.dropout),
            self.dropout)

        self.project_512 = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(128 * 8, self.dropout),
            self.dropout)

        self.project_1024 = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(256 * 8, self.dropout),
            self.dropout)

        self.project_2048 = hyper_layer.NormBlock(
            hyper_layer.Feed_Forward_Network(512 * 8, self.dropout),
            self.dropout)

        self.stage_norm_64 = hyper_layer.LayerNorm()
        self.stage_norm_256 = hyper_layer.LayerNorm()
        self.stage_norm_512 = hyper_layer.LayerNorm()
        self.stage_norm_1024 = hyper_layer.LayerNorm()
        self.stage_norm_2048 = hyper_layer.LayerNorm()

        # self.fianl_self_att = hyper_layer.NormBlock(
        #     hyper_layer.SelfAttention(
        #         num_heads=self.num_heads, num_units=1024,
        #         dropout=self.dropout), self.dropout)
        # self.final_ffn = hyper_layer.NormBlock(
        #     hyper_layer.Feed_Forward_Network(1024 * 4, self.dropout),
        #     self.dropout)
        self.encoder_output = hyper_layer.LayerNorm()

        self.init_conv = tf.keras.layers.Conv2D(
            16, (7, 7),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name='conv1')
        self.init_batchNorm = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name='bn_conv1')
        self.init_pool = tf.keras.layers.MaxPooling2D((2, 4),
                                                      strides=(2, 4),
                                                      padding='same')
        self.init_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, [1, 2]), name='reduce_mean')
        self.conv_block_256_a = hyper_layer.ResnetBlock2D(
            3, [16, 16, 64], stage=2, block='a', mode='conv')
        self.identity_block_256_b = hyper_layer.ResnetBlock2D(
            3, [16, 16, 64], stage=2, block='b')
        self.identity_block_256_c = hyper_layer.ResnetBlock2D(
            3, [16, 16, 64], stage=2, block='c')
        self.block_256_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, [1, 2]), name='reduce_mean_256')

        self.conv_block_512_a = hyper_layer.ResnetBlock2D(
            3, [32, 32, 128], strides=(2, 2), stage=3, block='a', mode='conv')
        self.identity_block_512_b = hyper_layer.ResnetBlock2D(
            3, [32, 32, 128], stage=3, block='b')
        self.identity_block_512_c = hyper_layer.ResnetBlock2D(
            3, [32, 32, 128], stage=3, block='c')
        self.identity_block_512_d = hyper_layer.ResnetBlock2D(
            3, [32, 32, 128], stage=3, block='d')
        self.block_512_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, [1, 2]), name='reduce_mean_512')

        self.conv_block_1024_a = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], strides=(2, 2), stage=4, block='a', mode='conv')
        self.identity_block_1024_b = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], stage=4, block='b')
        self.identity_block_1024_c = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], stage=4, block='c')
        self.identity_block_1024_d = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], stage=4, block='d')
        self.identity_block_1024_e = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], stage=4, block='e')
        self.identity_block_1024_f = hyper_layer.ResnetBlock2D(
            3, [64, 64, 256], stage=4, block='f')
        self.block_1024_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, [1, 2]), name='reduce_mean_1024')

        self.conv_block_2048_a = hyper_layer.ResnetBlock2D(
            3, [128, 128, 512],
            strides=(2, 2),
            stage=5,
            block='a',
            mode='conv')
        self.identity_block_2048_b = hyper_layer.ResnetBlock2D(
            3, [128, 128, 512], stage=5, block='a')
        self.identity_block_2048_c = hyper_layer.ResnetBlock2D(
            3, [128, 128, 512], stage=5, block='c')
        # self.identity_block_2048_d = hyper_layer.ResnetBlock2D(
        #     2, [512, 512, 2048], stage=5, block='d')
        self.block_2048_mean = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, [1, 2]), name='reduce_mean_2048')
        super(ResLinearEncoder, self).build(input_shape)

    def call(self, inputs, attention_bias, training):
        # inputs= inputs
        # inputs = Q
        # length = tf.shape(attention_bias)[-1]
        # attention_bias = tf.expand_dims(attention_bias, -2)
        x = inputs
        with tf.name_scope("stacked_encoder"):
            length = tf.shape(inputs)[1]
            # pos_encoding = conf_fn.get_position_encoding(length, 1024)
            x = tf.reshape(inputs, [-1, 32, 64, 3])
            res_paddings = tf.reduce_sum(tf.reshape(x, [-1, 32 * 64 * 3]), -1)
            res_paddings = tf.reshape(res_paddings, [-1, 1, 1, 1])
            # stage = 1
            x = self.init_conv(x)
            x = tf.nn.relu(self.init_batchNorm(x))
            mean_stage = self.init_mean(x)
            x = self.init_pool(x)
            x = self.norm_init(x) * res_paddings
            att_input = tf.reshape(mean_stage, [-1, length, 16])
            self_att = self.avg_self_attention_64(
                att_input, attention_bias, training=training)
            self_att = tf.reshape(self_att, [-1, 1, 1, 16])
            avg_self_att = tf.broadcast_to(
                tf.reshape(self_att, [-1, 1, 1, 16]),
                tf.shape(x)) * res_paddings

            m = tf.shape(x)[-3]
            n = tf.shape(x)[-2]
            att_input = tf.reshape(x, [-1, length, n * m, 16])
            self_att = self.self_attention_64(
                att_input, attention_bias, training=training)
            self_att = tf.reshape(self_att, [-1, n, m, 16])

            self_att = self.project_64_pixel(self_att) + self.project_64_avg(
                avg_self_att)
            self_att = self.project_64(self_att, training=training)
            # if training:
            #     self_att = tf.nn.dropout(self_att, rate=self.dropout)
            # x = x + self_att * res_paddings
            # x = tf.concat((x, self_att), -1)
            x = x + self_att
            # x = tf.concat(
            #     (x, tf.broadcast_to(self_att * res_paddings, tf.shape(x))), -1)
            # x = self.project_64(x, training=training) * res_paddings
            # x = self.project_64(x, training=training)
            x = self.stage_norm_64(x) * res_paddings

            # stage=2
            x = self.conv_block_256_a(x, training=training)
            x = self.identity_block_256_b(x, training=training)
            x = self.identity_block_256_c(x, training=training)
            mean_stage = self.block_256_mean(x)
            att_input = tf.reshape(mean_stage, [-1, length, 64])
            self_att = self.avg_self_attention_256(
                att_input, attention_bias, training=training)
            avg_self_att = tf.broadcast_to(
                tf.reshape(self_att, [-1, 1, 1, 64]),
                tf.shape(x)) * res_paddings
            x = self.norm_256(x) * res_paddings

            m = tf.shape(x)[-3]
            n = tf.shape(x)[-2]
            att_input = tf.reshape(x, [-1, length, n * m, 64])

            self_att = self.self_attention_256(
                att_input, attention_bias, training=training)
            self_att = tf.reshape(self_att, [-1, n, m, 64])

            # self_att = self_att + avg_self_att
            self_att = self.project_256_pixel(self_att) + self.project_256_avg(
                avg_self_att)
            self_att = self.project_256(self_att, training=training)
            # self_att = tf.reshape(self_att, [-1, 1, 1, 64])
            # if training:
            #     self_att = tf.nn.dropout(self_att, rate=self.dropout)

            # x = tf.concat(
            #     (x, tf.broadcast_to(self_att * res_paddings, tf.shape(x))), -1)
            # x = tf.concat((x, self_att), -1)
            x = x + self_att

            # x = self.project_256(x, training=training) * res_paddings
            # x = self.project_256(x, training=training)
            x = self.stage_norm_256(x) * res_paddings
            # stage = 3
            x = self.conv_block_512_a(x, training=training)
            x = self.identity_block_512_b(x, training=training)
            x = self.identity_block_512_c(x, training=training)
            x = self.identity_block_512_d(x, training=training)
            mean_stage = self.block_512_mean(x)
            att_input = tf.reshape(mean_stage, [-1, length, 128])
            self_att = self.avg_self_attention_512(
                att_input, attention_bias, training=training)
            avg_self_att = tf.broadcast_to(
                tf.reshape(self_att, [-1, 1, 1, 128]),
                tf.shape(x)) * res_paddings

            x = self.norm_512(x) * res_paddings
            m = tf.shape(x)[-3]
            n = tf.shape(x)[-2]
            att_input = tf.reshape(x, [-1, length, n * m, 128])
            self_att = self.self_attention_512(
                att_input, attention_bias, training=training)
            self_att = tf.reshape(self_att, [-1, n, m, 128])

            self_att = self.project_512_pixel(self_att) + self.project_512_avg(
                avg_self_att)
            self_att = self.project_512(self_att, training=training)
            # self_att = tf.reshape(self_att, [-1, 1, 1, 128])
            # if training:
            #     self_att = tf.nn.dropout(self_att, rate=self.dropout)
            # x = x + self_att * res_paddings

            # x = tf.concat(
            #     (x, tf.broadcast_to(self_att * res_paddings, tf.shape(x))), -1)
            # x = tf.concat((x, self_att), -1)
            x = x + self_att

            # x = self.project_512(x, training=training) * res_paddings
            # x = self.project_512(x,training=training)
            x = self.stage_norm_512(x) * res_paddings

            # stage=4
            x = self.conv_block_1024_a(x, training=training)
            x = self.identity_block_1024_b(x, training=training)
            x = self.identity_block_1024_c(x, training=training)
            x = self.identity_block_1024_d(x, training=training)
            x = self.identity_block_1024_e(x, training=training)
            x = self.identity_block_1024_f(x, training=training)
            mean_stage = self.block_1024_mean(x)
            att_input = tf.reshape(mean_stage, [-1, length, 256])
            self_att = self.avg_self_attention_1024(
                att_input, attention_bias, training=training)
            avg_self_att = tf.broadcast_to(
                tf.reshape(self_att, [-1, 1, 1, 256]),
                tf.shape(x)) * res_paddings

            x = self.norm_1024(x) * res_paddings
            m = tf.shape(x)[-3]
            n = tf.shape(x)[-2]
            att_input = tf.reshape(x, [-1, length, n * m, 256])

            self_att = self.self_attention_1024(
                att_input, attention_bias, training=training)
            self_att = tf.reshape(self_att, [-1, n, m, 256])
            # self_att = avg_self_att + self_att
            self_att = self.project_1024_pixel(
                self_att) + self.project_1024_avg(avg_self_att)
            self_att = self.project_1024(self_att, training=training)
            # self_att = tf.reshape(self_att, [-1, 1, 1, 256])
            # if training:
            #     self_att = tf.nn.dropout(self_att, rate=self.dropout)
            # x = x + self_att * res_paddings
            # x = tf.concat(
            #     (x, tf.broadcast_to(self_att * res_paddings, tf.shape(x))), -1)
            # x = tf.concat((x, self_att), -1)
            x = x + self_att
            # x = self.project_1024(x, training=training) * res_paddings
            # x = self.project_1024(x,training=training)
            x = self.stage_norm_1024(x) * res_paddings

            # stage=5
            x = self.conv_block_2048_a(x, training=training)
            x = self.identity_block_2048_b(x, training=training)
            x = self.identity_block_2048_c(x, training=training)
            mean_stage = self.block_2048_mean(x)
            att_input = tf.reshape(mean_stage, [-1, length, 512])
            self_att = self.avg_self_attention_2048(
                att_input, attention_bias, training=training)
            avg_self_att = tf.broadcast_to(
                tf.reshape(self_att, [-1, 1, 1, 512]),
                tf.shape(x)) * res_paddings

            x = self.norm_2048(x) * res_paddings
            m = tf.shape(x)[-3]
            n = tf.shape(x)[-2]
            att_input = tf.reshape(x, [-1, length, n * m, 512])

            self_att = self.self_attention_2048(
                att_input, attention_bias, training=training)
            # self_att = tf.broadcast_to(
            #     tf.reshape(self_att, [-1, 1, 1, 512]),
            #     tf.shape(x)) * res_paddings
            self_att = tf.reshape(self_att, [-1, n, m, 512])

            # self_att = self_att + avg_self_att
            self_att = self.project_2048_pixel(
                self_att) + self.project_2048_avg(avg_self_att)
            self_att = self.project_2048(self_att, training=training)
            # self_att = tf.reshape(self_att, [-1, 1, 1, 512])
            # if training:
            #     self_att = tf.nn.dropout(self_att, rate=self.dropout)
            # x = x + self_att * res_paddings
            # x = tf.concat(
            #     (x, tf.broadcast_to(self_att * res_paddings, tf.shape(x))), -1)
            # x = tf.concat((x, self_att), -1)
            x = x + self_att

            # x = self.project_2048(x, training=training) * res_paddings
            x = self.stage_norm_2048(x) * res_paddings
            # x = x + pos_encoding
            # x = self.fianl_self_att(x, attention_bias, training=training)
            x = self.project_final(x, training=training)
            x = tf.reshape(x, [-1, length, self.num_units])
        return self.encoder_output(x)

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
