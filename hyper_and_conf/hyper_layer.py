# encoder=utf8
import tensorflow as tf
from tensorflow.python.keras import regularizers
L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, num_units, num_heads, dropout):
        """Initialize Attention.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(num_units, num_heads))

        super(Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="q")
        self.k_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="k")
        self.v_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="v")
        self.output_dense_layer = tf.keras.layers.Dense(
            self.num_units, use_bias=False, name="output_transform")
        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.
    Args:
      x: A tensor with shape [batch_size, length, num_units]
    Returns:
      A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
    """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.num_units // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
    Args:
      x: A tensor [batch_size, num_heads, length, num_units/num_heads]
    Returns:
      A tensor with shape [batch_size, length, num_units]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(
                x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.num_units])

    def call(self, x, y, bias, training, cache=None):
        """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, num_units]
      y: a tensor with shape [batch_size, length_y, num_units]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, num_units]
    """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.num_units // self.num_heads)
        q *= depth**-0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if training:
            weights = tf.nn.dropout(weights, rate=self.dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, num_units]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, training, cache=None):
        return super(SelfAttention, self).call(x, x, bias, training, cache)


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, num_units, pad_id, name="embedding"):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      num_units: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.
    """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.pad_id = pad_id

    def build(self, input_shape):
        self.shared_weights = self.add_variable(
            shape=[self.vocab_size, self.num_units],
            dtype="float32",
            name="shared_weights",
            initializer=tf.random_normal_initializer(
                mean=0., stddev=self.num_units**-0.5))
        super(EmbeddingSharedWeights, self).build(input_shape)
        # self.build = True

    def call(self, inputs):
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.float32)
        embeddings = tf.gather(self.shared_weights, inputs)
        # embeddings = tf.nn.embedding_lookup(self.shared_weights, inputs)
        embeddings *= tf.expand_dims(mask, -1)
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.num_units**0.5

        return embeddings

    def linear(self, inputs):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, num_units]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.num_units])
        logits = tf.matmul(inputs, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            'vocab_size': self.vocab_size,
            'num_units': self.num_units,
            'pad_id': self.pad_id
        }
        # config.update(c)
        return c


class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
        mode:
            add: ln(x) + x
            norm: ln(x)
    """

    def __init__(self,
                 epsilon=1e-6,
                 gamma_initializer="ones",
                 beta_initializer="zeros"):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gamma_kernel = self.add_variable(
            shape=(input_dim),
            name="gamma",
            initializer=self.gamma_initializer)
        self.beta_kernel = self.add_variable(
            shape=(input_dim), name="beta", initializer=self.beta_initializer)
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs, training=False):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)
        output = self.gamma_kernel * normalized + self.beta_kernel
        return output

    def get_config(self):
        # config = super(LayerNorm, self).get_config()
        c = {'epsilon': self.epsilon}
        # config.update(c)
        return c


class NormBlock(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, dropout, add_mode=True):
        super(NormBlock, self).__init__()
        self.layer = layer
        # self.num_units = num_units
        self.dropout = dropout
        self.add_mode = add_mode

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNorm()
        super(NormBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, 'add_mode': self.add_mode}

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.dropout)
        return y + x


class Feed_Forward_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, num_units, dropout):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
        super(Feed_Forward_Network, self).__init__()
        self.num_units = num_units
        self.dropout = dropout

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.num_units,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = tf.keras.layers.Dense(
            out_dim, use_bias=True, name="output_layer")
        super(Feed_Forward_Network, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "dropout": self.dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_dense_layer(output)

        return output


class SequenceResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout):
        self.dropout = dropout
        self.filters = num_units
        super(SequenceResnetBlock, self).__init__('res_block')

    def build(self, input_shape):
        # self.filters = tf.shape(input_shape)[-1]
        self.bottel_fillters = int(self.filters / 4)
        self.shortcut_projection = tf.keras.layers.Conv1D(
            1,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.pre_norm = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1a = tf.keras.layers.Conv1D(
            self.bottel_fillters,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.bn1a = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1b = tf.keras.layers.Conv1D(
            self.bottel_fillters,
            3,
            padding='SAME',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY))
        self.bn1b = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )

        self.conv1c = tf.keras.layers.Conv1D(
            self.filters,
            1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        )
        self.bn1c = tf.keras.layers.BatchNormalization(
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )
        super(SequenceResnetBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, 'filters': self.filters}

    def call(self, inputs, post_norm=True, training=False):
        with tf.name_scope('res_block'):
            # shortcut = self.shortcut_projection(inputs)Vj
            # input_tensor = self.pre_norm(inputs)
            # input_tensor = tf.nn.relu(input_tensor)
            shortcut = inputs
            x = self.conv1a(inputs)
            x = self.bn1a(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv1b(x)
            x = self.bn1b(x, training=training)
            x = tf.nn.relu(x)

            x = self.conv1c(x)
            if post_norm:
                x = self.bn1c(x, training=training)
                # x = tf.keras.layers.BatchNormalization()(x)
                if training:
                    x = tf.nn.dropout(x, rate=self.dropout)
                x = x + shortcut
                return tf.nn.relu(x)
            else:
                if training:
                    x = tf.nn.dropout(x, rate=self.dropout)
                x = x + shortcut
                return x


class StackedSeqResBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, dropout, layers):
        self.num_units = num_units
        self.dropout = dropout
        self.layers = layers
        super(StackedSeqResBlock, self).__init__(name='StackedSeqResBlock')

    def get_config(self):
        return {'dropout': self.dropout, 'layer': self.layers}

    def build(self, input_shape):
        self.stacked_block = []
        for i in range(self.layers):
            self.stacked_block.append(
                SequenceResnetBlock(self.num_units, self.dropout))
        super(StackedSeqResBlock, self).build(input_shape)

    def call(self, inputs, post_norm=True, training=False):
        for index, layer in enumerate(self.stacked_block):
            with tf.name_scope('res_layer_%d' % index):
                inputs = layer(inputs, post_norm=post_norm, training=training)

        return inputs


# class ResnetIdentityBlock(tf.keras.layers.Layer):
#     def __init__(self, kernel_size, filters):
#         self.filters1, self.filters2, self.filters3 = filters
#         self.kernel_size = kernel_size
#         self.filters = filters
#         super(ResnetIdentityBlock, self).__init__('res_block')
#
#     def build(self, input_shape):
#         self.conv1a = tf.keras.layers.Conv1D(self.filters1, 1)
#         self.bn1a = tf.keras.layers.BatchNormalization()
#
#         self.conv1b = tf.keras.layers.Conv1D(
#             self.filters2, self.kernel_size, padding='same')
#         self.bn1b = tf.keras.layers.BatchNormalization()
#
#         self.conv1c = tf.keras.layers.Conv1D(self.filters3, 1)
#         self.bn1c = tf.keras.layers.BatchNormalization()
#         super(ResnetIdentityBlock, self).build(input_shape)
#
#     def get_config(self):
#         return {"kernel_size": self.kernel_size, 'filters': self.filters}
#
#     def call(self, input_tensor, training=False):
#         x = self.conv1a(input_tensor)
#         x = self.bn1a(x, training=training)
#         x = tf.nn.relu(x)
#
#         x = self.conv1b(x)
#         x = self.bn1b(x, training=training)
#         x = tf.nn.relu(x)
#
#         x = self.conv1c(x)
#         x = self.bn1c(x, training=training)
#
#         x += input_tensor
#         return tf.nn.relu(x)
