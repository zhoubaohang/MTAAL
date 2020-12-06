import os
import json
import argparse
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from collections import Counter
from utils import DATASET, BASEPATH

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['ncbi', 'cdr'], default='ncbi', type=str)
parser.add_argument('--wordembedding', choices=['word2vec', 'glove'], default='word2vec', type=str)
args = parser.parse_args()

CHARS = []
VOCABS = []
NER_TAG = []
NEN_TAG = []

WORD_EMBEDDINGS = {
    'word2vec': 'pretrain/GoogleNews-vectors-negative300.bin.gz',
    'glove': 'pretrain/glove300d.txt'
}

collect_token = lambda x, f: [w for w, i in iter(Counter(x).items()) if i >= f]

def collect_embeddings(vocabs, w2v_file):
    from gensim.models.keyedvectors import KeyedVectors
    if 'bin' in w2v_file:
        w2v = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    else:
        w2v = KeyedVectors.load_word2vec_format(w2v_file)
    embedding_dim = w2v.vector_size
    vb_size = len(vocabs)
    cnt = 0
    init_w = np.random.uniform(low=-0.25/np.sqrt(embedding_dim),
                               high=0.25/np.sqrt(embedding_dim),
                               size=(vb_size, embedding_dim))
    for idx in trange(vb_size, ascii=True):
        word = vocabs[idx].lower()
        if word in w2v:
            cnt += 1
            weight = w2v[word]
            init_w[idx] = weight
    print(f"In word2vec nums: {cnt}/{vb_size}")
    return init_w, embedding_dim

def collect_chars(vocabs):
    chars = []
    for w in vocabs:
        chars += [i for i in w.lower()]
    return collect_token(chars, 1)

if __name__ == '__main__':

    for k, v in DATASET[args.dataset]['to'].items():
        with open(f"{BASEPATH}{v}", 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.strip()
                if line:
                    word, ner, nen = line.split('\t')
                    nen = nen.split('|')[0]
                    VOCABS.append(word.lower())
                    NER_TAG.append(ner)
                    NEN_TAG.append(nen)

    # collect vocabulary
    VOCABS = collect_token(VOCABS, 1)
    NER_TAG = collect_token(NER_TAG, 1)
    NEN_TAG = collect_token(NEN_TAG, 1)
    VOCABS.insert(0, "UNK")
    VOCABS.insert(0, "GO")

    # collect chars
    CHARS = collect_chars(VOCABS)
    CHARS.insert(0, "unk")
    CHARS.insert(0, "pad")

    print(f"words:{len(VOCABS)} ner tags:{len(NER_TAG)} nen tags:{len(NEN_TAG)}")

    w2v, embedding_size = collect_embeddings(VOCABS, WORD_EMBEDDINGS[args.wordembedding])

    print("saving vocabs........")
    with open(f"{BASEPATH}{DATASET[args.dataset]['vocab']}", 'w', encoding='utf-8') as fp:
        json.dump({
            'chars': CHARS,
            'vocabs': VOCABS,
            'ner_tag': NER_TAG,
            'nen_tag': NEN_TAG,
            'w2v': w2v.tolist(),
            'embedding_size': embedding_size
        }, fp, ensure_ascii=False)

    print("--------DONE--------!")