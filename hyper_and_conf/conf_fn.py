# encoding=utf8
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib

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
        learning_rate = 0.1
        # Apply linear warmup
        learning_rate *= min(1, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= (1 / np.sqrt(max(step, warmup_steps)))

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


def onehot_loss_function(true,
                         pred,
                         mask_id=0,
                         smoothing=0.1,
                         vocab_size=24000):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """

    # mask = 1 - tf.cast(tf.equal(true, mask_id), tf.float32)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=pred, labels=true) * mask
    # return tf.reduce_mean(loss)
    logits, labels = pad_tensors_to_same_length(pred, true)
    # Calculate smoothing cross entropy
    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / tf.cast(
        vocab_size - 1, dtype=tf.float32)
    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    normalizing_constant = -(confidence * tf.math.log(confidence) + tf.cast(
        vocab_size - 1, dtype=tf.float32) * low_confidence *
                             tf.math.log(low_confidence + 1e-20))
    xentropy -= normalizing_constant

    weights = tf.cast(tf.not_equal(labels, mask_id), dtype=tf.float32)
    xentropy *= weights
    loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(
        input_tensor=weights)
    return loss


# lr test
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0, 15000, 1)
f = lambda lr,unit,step,warmup: get_learning_rate(lr,unit,step,warmup)
s = [f(2,1024,s,3000) for s in t]
fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='iteration', ylabel='lr',
       title='model learning rate')
ax.grid()

fig.savefig("lr.png")
plt.show()
