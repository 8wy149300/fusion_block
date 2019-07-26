# encode=utf8
# author barid
import tensorflow as tf
from hyper_and_conf import hyper_train, conf_metrics
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd + "/corpus")
sys.path.insert(1, cwd)
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus_fr2eng')
# sys.path.insert(1, '/Users/barid/Documents/workspace/alpha/transformer_nmt')
# device = ["/device:CPU:0", "/device:GPU:0", "/device:GPU:1"]
DATA_PATH = sys.path[0]
SYS_PATH = sys.path[1]
# src_data_path = DATA_PATH + "/europarl-v7.fr-en.en"
# tgt_data_path = DATA_PATH + "/europarl-v7.fr-en.fr"
# model = tf.keras.models.load_model("model_checkpoint/model.05.hdf5",custom_objects={'Daedalus':core_lip_main.Daedalus})
import core_model_initializer as init


# tf.keras.backend.set_session(tf.compat.v1.Session(config=config))
# gpu = init.get_available_gpus()
# # set session config
# metrics = init.get_metrics()
def get_vgg():
    # with tf.device("/cpu:0"):
    if tf.io.gfile.exists('pre_train/vgg16_pre_all'):
        vgg16 = tf.keras.models.load_model('pre_train/vgg16_pre_all')
    else:
        vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet')
    return vgg16


with tf.device("/cpu:0"):

    train_data = init.test_input()
    # X_Y = init.train_input(True)
    # val_x, val_y = init.val_input()
    # train_model = init.train_model()
    train_model = init.test_model()
    data_manager = init.data_manager
    # train_model.load_weights(tf.train.latest_checkpoint("./model_checkpoint/"))
    # train_model.load_weights("./model_checkpoint/model.01.ckpt")
    # with strategy.scope():
    hp = init.get_hp()
    # dataset
    # step
    train_step = 500
    # val_step = init.get_val_step()
    # get train model
    # bleu = hyper_train.Approx_BLEU_Metrics()
    # optimizer
    optimizer = init.get_optimizer()
    callbacks = init.get_callbacks()
    optimizer = tf.compat.v1.train.AdamOptimizer()
    pred = []
    for index, [x, y] in enumerate(train_data):
        import pdb; pdb.set_trace()
        # pred = initial_ids = tf.zeros([tf.shape(x[0])[0],1], dtype=tf.int64)
        #
        # # with tf.GradientTape() as tape:
        # # variables = train_model.trainable_variables
        # # import pdb; pdb.set_trace()
        # for i in range(10):
        #     logits = train_model((x[0], pred))
        #     initial_ids = tf.argmax(logits, -1)
        #     print(initial_ids)
        # #     pred = tf.concat((pred, initial_ids), -1)
        # # print(pred[:, 1:])
        # print("label")
        # print(y)


        ids = train_model(x)
        # import pdb; pdb.set_trace()
        print(ids)
        print("********************")
        for i in range(len(y)):
            print(ids[i])
            # print(data_manager.decode(ids[i]))
            # print("$$$$$")
            print(y[i])
            # print(data_manager.decode(y[i]))
            print("@@@@@@@@@@")
        print("#####################")
        # loss_v = conf_metrics.onehot_loss_function(y, logits, vocab_size=14000)
        # g = tape.gradient(loss_v, variables)
        # print(loss_v)
        # loss_v = loss(y,logits)
        # bleu_v = bleu(y,logits)
        # print(loss_v)
        # print(bleu_v)
        # grads = tape.gradient(loss_v, variables)p tf.reduce_sum(enc,-1)
        # try:
        #     checker = grads.index(None)
        #     print(variables[checker])
        # except Exception:
        #     pass
        # optimizer.apply_gradients(
        #     zip(grads, variables),
        #     global_step=tf.compat.v1.train.get_or_create_global_step())
        # break
        if index > 5:
            break
