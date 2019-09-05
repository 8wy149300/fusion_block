# encoding=utf8
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
import math
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'CPU'])


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])


def cpus_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def gpus_device():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_position_encoding(length,
                          hidden_size,
                          min_timescale=1.0,
                          max_timescale=1.0e4):
    """Return positional encoding.
  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_learning_rate(learning_rate,
                      hidden_size,
                      step=1,
                      learning_rate_warmup_steps=16000):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.compat.v1.name_scope("learning_rate"):
        warmup_steps = float(learning_rate_warmup_steps)
        # try:
        #     step = tf.to_float(tf.train.get_or_create_global_step())
        # except Exception:
        #     step = 1
        step = max(1.0, step)
        # step = max(1.0, 10)

        learning_rate *= (hidden_size**-0.5)
        # print(learning_rate)
        # learning_rate = 0.1
        # Apply linear warmup
        learning_rate *= min(1, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= (0.5 / np.sqrt(max(step, warmup_steps)))

        # # Create a named tensor that will be logged using the logging hook.
        # # The full name includes variable and names scope. In this case, the name
        # # is model/get_train_op/learning_rate/learning_rate
        # ratio = 10
        # cut_frac = 0.4
        # cut = int(3000 * cut_frac)
        #
        # if step < cut:
        #     p = step / cut
        # else:
        #     p = 1 - ((step - cut) / (cut * (ratio - 1)))
        # learning_rate = 0.01 * (1 + p * (ratio - 1)) / ratio
        return learning_rate


def pad_tensors_to_same_length(x, y, pad_id=0):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = tf.shape(input=x)[1]
    y_length = tf.shape(input=y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(
        tensor=x,
        paddings=[[0, 0], [0, max_length - x_length], [0, 0]],
        constant_values=pad_id)
    y = tf.pad(
        tensor=y,
        paddings=[[0, 0], [0, max_length - y_length]],
        constant_values=pad_id)
    return x, y
