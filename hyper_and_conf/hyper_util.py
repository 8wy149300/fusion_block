import math

import tensorflow as tf

_NEG_INF = -1e9


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.
  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  Args:
    length: int length of sequences in batch.
  Returns:
    float tensor of shape [1, 1, length, length]
  """
    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.
  Args:
    x: int tensor with any shape
    padding_value: int value that
  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_value), tf.float32)


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.
  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  Args:
    x: int tensor with shape [batch_size, length]
  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias
