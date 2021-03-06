# encoding=utf-8
# author barid
import tensorflow as tf
# tf.config.set_soft_device_placement(True)
# tf.config.gpu.set_per_process_memory_growth(0.99)
import core_model_initializer as init
from hyper_and_conf import conf_fn
import sys
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = (
#     "0,1"
# )  # specify which GPU(s) to be used# os.environ["CUDA_VISIBLE_DEVICES"] = str(init.get_available_gpus())
# src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
# tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"

# tf.debugging.set_log_device_placement(True)
# tf.compat.v1.disable_eager_execution()# tf.enable_eager_execution()
# tf.config.gpu.set_per_process_memory_fraction(0.7)

cwd = os.getcwd()
sys.path.insert(0, cwd + "/corpus")
sys.path.insert(1, cwd)
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# device = ["/device:CPU:0", "/device:GPU:0", "/device:GPU:1"]
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

# devices = tf.config.experimental_list_devices()

MODE = 'LIP'

# MODE = 'TRANSFORMER'


def _load_weights_if_possible(self, model, init_weight_path=None):
    if init_weight_path:
        tf.compat.v1.logging.info("Load weights: {}".format(init_weight_path))
        model.load_weights(init_weight_path)
    else:
        tf.compat.v1.logging.info(
            "Weights not loaded from path:{}".format(init_weight_path))


def main():
    num_gpus = conf_fn.get_available_gpus()
    if num_gpus == 0:
        devices = ["device:CPU:0"]
    else:
        devices = ["device:GPU:%d" % i for i in range(num_gpus)]
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    callbacks = init.get_callbacks()
    pre_model = init.train_model()  # train_model = init.test_model()
    text_model = init.model_structure(mode='text')
    # pre_model.load_weights(
    #     "/home/vivalavida/workspace/alpha/lip_read_dual_bleu1/model_checkpoint/model.05.ckpt"
    # )
    # text_model.load_weights(
    #     "/home/vivalavida/massive_data/lip_read_lm/model_checkpoint/model.05.ckpt"
    # )
    # # word_weights_0 = pre_model.layers[2].layers[0].get_weights()
    # word_weights_1 = pre_model.layers[2].layers[1].get_weights()
    # word_weights_2 = pre_model.layers[2].layers[2].get_weights()
    # text_weights_0 = text_model.layers[2].layers[0].get_weights()
    # text_weights_3 = text_model.layers[2].layers[3].get_weights()
    with strategy.scope():
        train_model = init.train_model()
        if MODE == 'LIP':
            train_dataset = init.train_input()
        else:
            train_dataset = init.train_Transformer_input()
        # pre_model.load_weights(
        #     tf.train.latest_checkpoint(
        #         "/home/vivalavida/massive_data/model_checkpoint"))

        # train_model.layers[2].layers[0].set_weights(word_weights_0)
        # train_model.layers[2].layers[1].set_weights(word_weights_1)
        # train_model.layers[2].layers[0].set_weights(text_weights_0)
        # train_model.layers[2].layers[3].set_weights(text_weights_3)
        # train_model.layers[2].layers[2].set_weights(word_weights_2)
        optimizer = init.get_optimizer()
        train_model.compile(optimizer=optimizer)
        train_model.summary()
        # train_dataset = tf.keras.applications.resnet50.preprocess_input(train_dataset)
        # print(train_dataset.take(1))
        # train_model.evaluate(train_dataset, verbose=1, callbacks=callbacks)
        # for i in range(1, 100):
        train_model.fit(
            train_dataset,
            epochs=100,
            verbose=1,
            callbacks=callbacks,
            shuffle=True,
        )

        train_model.save_weights("model_weights")


if __name__ == "__main__":
    main()
# import tensorflow.python.keras.engine.distributed_training_utils
# import tensorflow.python.keras.engine.training_arrays
