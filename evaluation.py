# encoding=utf-8
# author barid
import sys
import os

import tensorflow as tf

from hyper_and_conf import conf_metrics

tf.config.set_soft_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import core_model_initializer as init

cwd = os.getcwd()
sys.path.insert(0, cwd + "/corpus")
sys.path.insert(1, cwd)
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

def _load_weights_if_possible(self, model, init_weight_path=None):
    if init_weight_path:
        tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
        model.load_weights(init_weight_path)
    else:
        tf.compat.v1.logging.info(
            "Weights not loaded from path:{}".format(init_weight_path))


def main(small=False):
    callbacks = init.get_callbacks()
    loss = init.get_external_loss()
    train_model = init.test_model()
    print(tf.train.latest_checkpoint("./model_checkpoint/"))
    # train_model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
    optimizer = init.get_optimizer()
    train_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[conf_metrics.wer_fn, conf_metrics.bleu_fn])
    train_model.summary()
    train_dataset = init.val_input()
    train_model.evaluate(train_dataset, callbacks=callbacks)



if __name__ == "__main__":
    main(True)
# import tensorflow.python.keras.engine.distributed_training_utils
# import tensorflow.python.keras.engine.training_arrays
