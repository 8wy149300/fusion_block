# encoding=utf-8
import sys
from hyper_and_conf import hyper_param as hyperParam
from hyper_and_conf import hyper_train
import core_lip_main
import core_data_SRCandTGT
from tensorflow.python.client import device_lib
import tensorflow as tf
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
TRAIN_PATH = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2'
C = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2/main'
# with open(TRAIN_PATH + '/lr_train.txt', 'r') as f:
#     files = f.readlines()
# files = [C + '/' + f.rstrip() for f in files]
# with open(TRAIN_PATH + '/lr_test.txt', 'r') as f:
#     file = f.readlines()
# file = [C + '/' + f.split(' ')[0].rstrip() for f in file]
# files = files + file
# with open(TRAIN_PATH + '/val.txt', 'r') as f:
#     file = f.readlines()
# file = [C + '/' + f.rstrip() for f in file]
# files = files + file
# files = [f + '.txt' for f in files]
#
# src_data_path = files
# tgt_data_path = files
src_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]

tgt_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]
# TFRECORD = '/home/vivalavida/massive_data/lip_reading_TFRecord/tfrecord_word'
# TFRECORD = '/home/vivalavida/massive_data/sentence_lip_data_tfrecord_v3'
# TFRECORD = '/home/vivalavida/massive_data/sentence_lip_data_tfrecord_train_v2'
# TFRECORD = '/home/vivalavida/massive_data/sentence_lip_data_tfrecord_train'

TFRECORD = '/data'

import numpy as np

# TFRECORD = '/Users/barid/Documents/workspace/batch_data/sentence_lip_data_tfrecord_train'


def get_vgg(self):
    if tf.io.gfile.exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet')
    return vgg16


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


gpu = get_available_gpus()
TRAIN_MODE = 'large' if gpu > 0 else 'small'
hp = hyperParam.HyperParam(TRAIN_MODE, gpu=get_available_gpus())
PAD_ID = tf.cast(hp.PAD_ID, tf.int64)
with tf.device("/cpu:0"):
    # if tf.gfile.Exists('pre_train/vgg16_pre_all'):
    #     vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    # else:
    #     vgg16 = tf.keras.applications.vgg16.VGG16(
    #         include_top=True, weights='imagenet')
    data_manager = core_data_SRCandTGT.DatasetManager(
        src_data_path,
        tgt_data_path,
        batch_size=hp.batch_size,
        PAD_ID=hp.PAD_ID,
        EOS_ID=hp.EOS_ID,
        # shuffle=hp.data_shuffle,
        shuffle=hp.data_shuffle,
        max_length=hp.max_sequence_length,
        tfrecord_path=TFRECORD)

# train_dataset, val_dataset, test_dataset = data_manager.prepare_data()


def get_hp():
    return hp


def backend_config():
    config = tf.compat.v1.ConfigProto()
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    # # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.999
    # config.allow_soft_placement = True

    return config


def input_fn(flag="TRAIN"):
    if flag == "VAL":
        dataset = data_manager.get_raw_val_dataset()
    else:
        if flag == "TEST":
            dataset = data_manager.get_raw_test_dataset()
        else:
            if flag == "TRAIN":
                dataset = data_manager.get_raw_train_dataset()
            else:
                assert ("data error")
        # repeat once in case tf.keras.fit out range error
        # if get_available_gpus() > 0:
        # dataset = dataset.shuffle(
        #     hp.data_shuffle,
        #     reshuffle_each_iteration=True)

    return dataset


def pad_sample(dataset, seq2seq=False):
    if seq2seq:
        dataset = dataset.map(dataset_prepross_fn, num_parallel_calls=12)
        dataset = dataset.padded_batch(
            hp.batch_size,
            padded_shapes=(
                (
                    tf.TensorShape([320,
                                    None]),  # source vectors of unknown size
                    tf.TensorShape([256]),  # target vectors of unknown size
                ),
                tf.TensorShape([256])),
            padding_values=(
                (
                    tf.cast(
                        hp.PAD_ID, tf.float32
                    ),  # source vectors padded on the right with src_eos_id
                    PAD_ID
                    # target vectors padded on the right with tgt_eos_id
                ),
                PAD_ID),
            drop_remainder=True)

    else:

        dataset = dataset.padded_batch(
            hp.batch_size,
            padded_shapes=(
                tf.TensorShape([320, None]),  # source vectors of unknown size
                tf.TensorShape([300]),  # target vectors of unknown size
            ),
            padding_values=(
                tf.cast(
                    hp.PAD_ID, tf.float32
                ),  # source vectors padded on the right with src_eos_id
                PAD_ID
                # target vectors padded on the right with tgt_eos_id
            ),
            drop_remainder=True)

    return dataset


def get_train_step():
    return data_manager.get_train_size() // hp.batch_size


def get_val_step():
    return data_manager.get_val_size() // hp.batch_size


def get_test_step():
    return data_manager.get_test_size() // hp.batch_size


def dataset_prepross_fn(src, tgt, val=False):
    if val:
        return (src, 0), tgt
    return (src, tgt), tgt


def train_input(seq2seq=True):
    dataset = input_fn('TRAIN')
    dataset = pad_sample(dataset, seq2seq=seq2seq)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # if gpu > 0:
    #     for i in range(gpu):
    #         dataset = dataset.apply(
    #             tf.data.experimental.prefetch_to_device("/GPU:" + str(i)))
    # else:
    return dataset


def val_input(seq2seq=True):
    dataset = input_fn("VAL")
    dataset = pad_sample(dataset, seq2seq=seq2seq)
    return dataset


def model_structure(training=True):
    daedalus = core_lip_main.Daedalus(
        hp.max_sequence_length, hp.vocabulary_size, hp.embedding_size,
        hp.batch_size, hp.num_units, hp.num_heads, hp.num_encoder_layers,
        hp.num_decoder_layers, hp.dropout, hp.EOS_ID, hp.PAD_ID, hp.MASK_ID)
    img_input = tf.keras.layers.Input(
        shape=[None, 4096], dtype=tf.float32, name='VGG_features')
    tgt_input = tf.keras.layers.Input(
        shape=[None], dtype=tf.int64, name='tgt_input')
    output = daedalus((img_input, tgt_input), training=training)
    model = tf.keras.Model(inputs=(img_input, tgt_input), outputs=output)
    # if multi_gpu and gpu > 0:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=gpu)
    return model


def train_model():
    return model_structure(True)


def test_model():
    return model_structure(False)


def get_metrics():
    # evaluation metrics
    bleu = hyper_train.Approx_BLEU_Metrics(eos_id=hp.EOS_ID)
    accuracy = hyper_train.Padded_Accuracy(hp.PAD_ID)
    accuracy_topk = hyper_train.Padded_Accuracy_topk(k=10, pad_id=hp.PAD_ID)
    seq_accuracy = hyper_train.Padded_Seq_Accuracy(hp.PAD_ID)
    wer = hyper_train.Approx_WER_Metrics()
    # bleu = metrics.MeanMetricWrapper(conf_metrics.bleu_score, name='padded_accuracy_score')
    # accuracy = metrics.MeanMetricWrapper(conf_metrics.padded_accuracy_score, name='padded_accuracy_score')
    # accuracy_topk = metrics.MeanMetricWrapper(
    #     conf_metrics.padded_accuracy_score_topk, name='topk_score')
    return [bleu, wer, seq_accuracy, accuracy, accuracy_topk]


def get_optimizer():
    return tf.keras.optimizers.Adam()


def get_loss(training=True):
    return hyper_train.Onehot_CrossEntropy(hp.vocabulary_size)


def get_callbacks():
    LRschedule = hyper_train.Dynamic_LearningRate(hp.lr, hp.num_units,
                                                  hp.learning_warmup)
    TFboard = tf.keras.callbacks.TensorBoard(
        log_dir=hp.model_summary_dir,
        histogram_freq=10,
        write_images=True,
        update_freq=10)

    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        hp.model_checkpoint_dir + '/model.{epoch:02d}.ckpt',
        save_weights_only=True,
        verbose=1,
    )
    # BatchTime = hyper_train.BatchTiming()
    # SamplesPerSec = hyper_train.SamplesPerSec(hp.batch_size)
    # if get_available_gpus() > 0:
    #     CudaProfile = hyper_train.CudaProfile()
    #
    #     return [
    #         LRschedule, TFboard, TFchechpoint, BatchTime, SamplesPerSec,
    #         CudaProfile
    #     ]
    # else:
    return [LRschedule, TFboard, TFchechpoint]
