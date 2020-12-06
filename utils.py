BASEPATH = './dataset/'
DATASET = {
    'ncbi': {
        'from':{
            'train': 'NCBI/NCBItrainset_corpus.txt',
            'dev': 'NCBI/NCBIdevelopset_corpus.txt',
            'test': 'NCBI/NCBItestset_corpus.txt'
        },
        'to':{
            'train': 'NCBI/train.txt',
            'dev': 'NCBI/dev.txt',
            'test': 'NCBI/test.txt'
        },
        'vocab': 'NCBI/vocab.json'
    },
    'cdr': {
        'from':{
            'train': 'CDR/CDR_TrainingSet.PubTator.txt',
            'dev': 'CDR/CDR_DevelopmentSet.PubTator.txt',
            'test': 'CDR/CDR_TestSet.PubTator.txt'
        },
        'to':{
            'train': 'CDR/train.txt',
            'dev': 'CDR/dev.txt',
            'test': 'CDR/test.txt'
        },
        'vocab': 'CDR/vocab.json'
    }
}
import os
import json
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt
from evaluate import precision_recall_f1
from keras.losses import categorical_crossentropy
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

tf_one_hot = lambda x, size: tf.one_hot(tf.cast(x, tf.int32), size)
categorical_cross_entropy = lambda y_pred, y_true:\
    tf.reduce_mean(categorical_crossentropy(y_pred,\
        tf_one_hot(y_true, y_pred.get_shape().as_list()[-1])))

def calc_nen_f1(true, pred, real_len):
    y_real, pred_real = [], []
    for i in range(len(real_len)):
        y_real += true[i, -real_len[i]:].tolist()
        pred_real += pred[i, -real_len[i]:].tolist()

    prec = precision_score(y_real, pred_real, average='macro')
    reca = recall_score(y_real, pred_real, average='macro')
    f1 = f1_score(y_real, pred_real, average='macro')
    acc = accuracy_score(y_real, pred_real)

    return (prec, reca, f1, acc)

def calc_ner_f1(true, pred, real_len, tags=None):
    y_real, pred_real = [], []
    for i in range(len(real_len)):
        y_real.extend(true[i, -real_len[i]:].tolist())
        pred_real.extend(pred[i, -real_len[i]:].tolist())

    total = precision_recall_f1(y_real, pred_real, print_results=False)['__total__']
    prec = total['precision'] / 100.
    rec = total['recall'] / 100.
    f1 = total['f1'] / 100.

    return (prec, rec, f1)

def save_result(file_name, dataset_name, model_name, ner, nen):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            pass
    with open(file_name, 'r') as f:
        result = json.load(f) if os.path.getsize(file_name) else {}
        if dataset_name not in result:
            result[dataset_name] = {}
        result[dataset_name][model_name] = {
            'ner': ner, 'nen': nen
        }

    with open(file_name, 'w') as f:
        json.dump(result, f)

def statistic_data(sentences, tags):
    words = []
    entity = []
    n_entity = 0
    
    for i in range(len(sentences)):
        for w in sentences[i]:
            words.append(w)

        if 'B-' in ' '.join(tags[i]):
            entity.append(1)
        else:
            entity.append(0)
        
        for t in tags[i]:
            if 'B-' in t:
                n_entity += 1
    
    counter = Counter(words)
    cnt_word = len(counter)
    we = len(np.where(np.array(entity))[0])
    woe = len(entity) - we

    return {
        'words': list(counter.keys()),
        'cnt_word': cnt_word,
        'n_entity': n_entity,
        'we': we,
        'woe': woe
    }

def plot_to_pdf(path: str):
    '''
        A decorator for saving figures as pdf file
    '''
    def _deco(func):
        def __deco(*args, **kwargs):
            with PdfPages(path) as pdf:
                func(*args, **kwargs)
                pdf.savefig()
                plt.show()
                plt.close()
        return __deco
    return _deco
