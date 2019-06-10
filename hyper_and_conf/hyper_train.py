# encoding=utf8
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import hyper_and_conf.conf_metrics as conf_metrics
import hyper_and_conf.conf_fn as conf_fn
import numpy as np
from tensorflow.python.keras.losses import Loss


class Onehot_CrossEntropy(Loss):
    def __init__(self, vocab_size, mask_id=0, smoothing=0.1):

        super(Onehot_CrossEntropy, self).__init__(name="Onehot_CrossEntropy")
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.smoothing = smoothing

    def call(self, true, pred):
        batch_size = tf.shape(true)[0]
        true = tf.reshape(true, [batch_size, -1])
        loss = conf_metrics.onehot_loss_function(
            true=true,
            pred=pred,
            mask_id=self.mask_id,
            smoothing=self.smoothing,
            vocab_size=self.vocab_size)
        return loss


class CrossEntropy(tf.keras.layers.Layer):
    def __init__(self, vocab_size, label_smoothing):
        super(CrossEntropy, self).__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

    def build(self, input_shape):
        super(CrossEntropy, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        logits, targets = inputs[0], inputs[1]
        loss = conf_metrics.onehot_loss_function(
            targets,
            logits,
            smoothing=self.label_smoothing,
            vocab_size=self.vocab_size)
        self.add_loss(loss)
        return logits


class MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, vocab_size):
        super(MetricLayer, self).__init__()
        self.vocab_size = vocab_size
        self.metric_mean_fns = []

    def build(self, input_shape):
        """"Builds metric layer."""
        self.metric_mean_fns = [
            (tf.keras.metrics.Mean("approx_bleu"), conf_metrics.approx_bleu),
            (tf.keras.metrics.Mean("wer"), conf_metrics.wer_score),
            (tf.keras.metrics.Mean("accuracy"), conf_metrics.padded_accuracy),
            (tf.keras.metrics.Mean("accuracy_top5"),
             conf_metrics.padded_accuracy_top5),
            (tf.keras.metrics.Mean("accuracy_per_sequence"),
             conf_metrics.padded_sequence_accuracy),
        ]
        super(MetricLayer, self).build(input_shape)

    def get_config(self):
        return {"vocab_size": self.vocab_size}

    def call(self, inputs):
        logits, targets = inputs[0], inputs[1]
        # TODO(guptapriya): Remove this check when underlying issue to create metrics
        # with dist strat in cross replica context is fixed.
        if tf.distribute.has_strategy(
        ) and not tf.distribute.in_cross_replica_context():
            for mean, fn in self.metric_mean_fns:
                m = mean(*fn(logits, targets))
                self.add_metric(m)
        # else:
        #     for mean, fn in self.metric_mean_fns:
        #         m = mean(*fn(logits, targets))
        #         self.add_metric(m)
        return logits


class Dynamic_LearningRate(Callback):
    def __init__(self,
                 init_lr,
                 num_units,
                 learning_rate_warmup_steps,
                 verbose=0):
        super(Dynamic_LearningRate, self).__init__()
        self.init_lr = init_lr
        self.num_units = num_units
        self.learning_rate_warmup_steps = learning_rate_warmup_steps
        self.verbose = verbose
        self.sess = tf.compat.v1.keras.backend.get_session()
        self._total_batches_seen = 0
        self.current_lr = 0

    def on_train_begin(self, logs=None):
        self.current_lr = conf_fn.get_learning_rate(
            self.init_lr, self.num_units, self._total_batches_seen,
            self.learning_rate_warmup_steps)
        lr = float(self.current_lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nStart  learning ' 'rate from %s.' % (lr))

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            self.current_lr = conf_fn.get_learning_rate(
                self.init_lr, self.num_units, self._total_batches_seen,
                self.learning_rate_warmup_steps)
        except Exception:  # Support for old API for backward compatibility
            self.current_lr = self.init_lr
        lr = float(self.current_lr)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            tf.compat.v1.logging.info('\nEpoch %05d: Changing  learning '
                                      'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        # path = os.path.join("model_summary", "train")
        # writer = summary_ops_v2.create_file_writer(path)
        # with summary_ops_v2.always_record_summaries():
        #     with writer.as_default():
        #         summary_ops_v2.scalar(
        #             "lr", self.current_lr, step=self._total_batches_seen)
        self._total_batches_seen += 1
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    # def _log_lr(logs, prefix,step):
