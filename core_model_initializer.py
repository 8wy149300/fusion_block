# encoding=utf-8
import sys
from hyper_and_conf import hyper_param as hyperParam
from hyper_and_conf import hyper_train, hyper_optimizer
import core_lip_main
import core_data_SRCandTGT
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
# TRAIN_PATH = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2'
# C = '/home/vivalavida/massive_data/lip_reading_data/sentence_level_lrs2/main'
src_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]

tgt_data_path = [DATA_PATH + "/corpus/lip_corpus.txt"]
# TFRECORD = '/home/vivalavida/massive_data/lip_reading_TFRecord/tfrecord_word'
# TFRECORD = '/home/vivalavida/massive_data/sentence_lip_data_tfrecord_train_v1'
TFRECORD = '/home/vivalavida/massive_data/lr_train'

TFRECORD = '/home/wonderwall/data/lr_train'

# TFRECORD = '/home/vivalavida/massive_data/fc1'

# TFRECORD = '/data'

# TFRECORD = '/Users/barid/Documents/workspace/batch_data/'


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
PAD_ID_int64 = tf.cast(hp.PAD_ID, tf.int64)
PAD_ID_float32 = tf.cast(hp.PAD_ID, tf.float32)

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
    return dataset


def map_data_for_feed_pertunated(x, y):
    return ((x, randomly_pertunate_input(y)), y)


def map_data_for_feed(x, y):
    return ((x, y), y)


def randomly_pertunate_input(x):
    determinater = np.random.randint(10)
    if determinater > 3:
        return x
    else:
        index = np.random.randint(2, size=(1, 80))
        x = x * index
    return x


def pad_sample(dataset, batch_size):
    # dataset = dataset.shuffle(200000, reshuffle_each_iteration=True)
    dataset = dataset.padded_batch(
        hp.batch_size,
        (
            [200, None],  # source vectors of unknown size
            [80]),  # target vectors of unknown size
        (PAD_ID_float32, PAD_ID_int64),
        drop_remainder=True)

    return dataset


def dataset_prepross_fn(src, tgt, val=False):
    if val:
        return (src, 0), tgt
    return (src, tgt), tgt


def train_input(seq2seq=True, pertunate=False):

    dataset = input_fn('TRAIN')
    # dataset = dataset.shuffle(100000)
    dataset = pad_sample(dataset, batch_size=hp.batch_size)

    if pertunate:
        dataset = dataset.map(map_data_for_feed_pertunated)
    else:
        dataset = dataset.map(map_data_for_feed)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def test_input(seq2seq=True, pertunate=False):

    dataset = input_fn('TRAIN')
    # dataset = dataset.shuffle(100000)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def val_input(seq2seq=True):
    dataset = input_fn("TRAIN")
    dataset = pad_sample(dataset, 4)
    # dataset = dataset.map(map_data_for_val)
    return dataset


def get_external_loss():
    return hyper_train.Onehot_CrossEntropy(hp.vocabulary_size)


def model_structure(training=True):
    if training:
        img_input = tf.keras.layers.Input(
            shape=[200, 4096], dtype=tf.float32, name='VGG_features')
        tgt = tf.keras.layers.Input(
            shape=[None], dtype=tf.int64, name='target_text')
        # metric = hyper_train.MetricLayer(hp.vocabulary_size)
        # loss = hyper_train.CrossEntropy(hp.vocabulary_size, 0.1)
        daedalus = core_lip_main.Daedalus(
            hp.max_sequence_length, hp.vocabulary_size, hp.embedding_size,
            hp.batch_size, hp.num_units, hp.num_heads, hp.num_encoder_layers,
            hp.num_decoder_layers, hp.dropout, hp.EOS_ID, hp.PAD_ID,
            hp.MASK_ID)
        logits = daedalus([img_input, tgt], training=training)
        logits = hyper_train.MetricLayer(hp.vocabulary_size)([logits, tgt])
        # logits = hyper_train.CrossEntropy(hp.vocabulary_size,
        #                                   0.1)([logits, tgt])
        logits = tf.keras.layers.Lambda(lambda x: x, name="logits")(logits)

        model = tf.keras.Model(inputs=[img_input, tgt], outputs=logits)
    else:
        img_input = tf.keras.layers.Input(
            shape=[200, 4096], dtype=tf.float32, name='VGG_features')
        daedalus = core_lip_main.Daedalus(
            hp.max_sequence_length, hp.vocabulary_size, hp.embedding_size,
            hp.batch_size, hp.num_units, hp.num_heads, hp.num_encoder_layers,
            hp.num_decoder_layers, hp.dropout, hp.EOS_ID, hp.PAD_ID,
            hp.MASK_ID)
        # metric = hyper_train.MetricLayer(hp.vocabulary_size)
        # loss = hyper_train.CrossEntropy(hp.vocabulary_size, 0.1)
        ret = daedalus([img_input], training=training)
        outputs, scores = ret["outputs"], ret["scores"]
        model = tf.keras.Model(img_input, outputs)
    # if multi_gpu and gpu > 0:
    #     model = tf.keras.utils.multi_gpu_model(model, gpus=gpu)
    return model


def train_model():
    return model_structure(training=True)


def test_model():
    return model_structure(training=False)


def get_optimizer():
    return tf.keras.optimizers.Adam(beta_1=0.1, beta_2=0.98, epsilon=1.0e-9)
    # return hyper_optimizer.LazyAdam(beta_1=0.1, beta_2=0.98, epsilon=1.0e-9)


def get_callbacks():
    lr_fn = hyper_optimizer.LearningRateFn(hp.lr, hp.num_units,
                                           hp.learning_warmup)
    LRschedule = hyper_optimizer.LearningRateScheduler(lr_fn, 0)
    TFboard = tf.keras.callbacks.TensorBoard(
        log_dir=hp.model_summary_dir,
        write_grads=True,
        histogram_freq=100,
        write_images=True,
        update_freq=100)
    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        hp.model_checkpoint_dir + '/model.{epoch:02d}.ckpt',
        save_weights_only=True,
        verbose=1,
    )
    return [LRschedule, TFboard, TFchechpoint]
