# encode=utf8

# author barid
import tensorflow as tf
from hyper_and_conf.conf_metrics import compute_wer, compute_bleu, token_trim
import sys
import os
cwd = os.getcwd()
sys.path.insert(0, cwd + '/corpus')
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

with tf.device("/cpu:0"):
    val_data = init.val_input()
    val_model = init.test_model()
    data_parser = init.data_manager
    for i in range(5, 100):
        if i < 10:
            index = '0' + str(i)
        else:
            index = str(i)
        ckpt_path = 'model_checkpoint/model.' + index + '.ckpt'
        val_model.load_weights(ckpt_path)
        hyp = []
        re = []
        bleu = 0
        wer = 0
        n = 0
        for index, (x, y) in enumerate(val_data):

            logits = val_model(x)
            # hyp.append(data_parser.decode(logits[0]))
            # re.append(data_parser.decode(y[0])[:-8])
            # # pre = tf.squeeze(logits)
            # bleu += compute_bleu(re, hyp)
            # wer += compute_wer(hyp, re)
            # n += 1
            break
        print('\n')
        print("Evaluation on checkpoint:{0}".format(ckpt_path))
        # print('Bleu:{0}, WER:{1}'.format(bleu/n, wer/n))
