# encoding=utf-8
# author barid
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used# os.environ["CUDA_VISIBLE_DEVICES"] = str(init.get_available_gpus())
# src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
# tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)
# tf.debugging.set_log_device_placement(True)
# tf.compat.v1.disable_eager_execution()# tf.enable_eager_execution()
# tf.config.gpu.set_per_process_memory_growth(True)
# tf.config.gpu.set_per_process_memory_fraction(0.7)
tf.config.set_soft_device_placement(True)
import core_model_initializer as init
cwd = os.getcwd()
sys.path.insert(0, cwd + '/corpus')
sys.path.insert(1, cwd)
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# device = ["/device:CPU:0", "/device:GPU:0", "/device:GPU:1"]
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]

# devices = tf.config.experimental_list_devices()


def main():
    gpu = init.get_available_gpus()
    # device = [
    #     "/device:GPU:0", "/device:GPU:1", "/device:GPU:2", "/device:GPU:3",
    #     "/device:GPU:4", "/device:GPU:5", "/device:GPU:6", "/device:GPU:7"
    # ]
    device = [
        "/device:GPU:0", "/device:GPU:1"
    ]
    if gpu > 0:
        # device = tf.config.experimental_list_devices()
        strategy = tf.distribute.MirroredStrategy(devices=device)
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        device = init.cpus_device()
        strategy = tf.distribute.MirroredStrategy(devices=device)

    # val_dataset = init.val_input()
    with strategy.scope():
        # with tf.device("/cpu:0"):
        train_dataset = init.train_input()
        train_model = init.train_model()
        # train_dataset = strategy.make_dataset_iterator(train_dataset)
        metrics = init.get_metrics()
        # hp = init.get_hp()
        # train_step = 550
        optimizer = init.get_optimizer()
        loss = init.get_loss()
        callbacks = init.get_callbacks()
        # if gpu > 0:
        #     train_model = tf.keras.utils.multi_gpu_model(train_model, 2)
        train_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
    train_model.summary()
    # train_model.evaluate(
    #     val_dataset,
    #     # epochs=100,
    #     # validation_data=val_dataset,
    #     verbose=1,
    #     callbacks=callbacks)

    train_model.fit(train_dataset, epochs=100, verbose=1, callbacks=callbacks)

    train_model.save_weights("model_weights")


if __name__ == '__main__':
    main()
