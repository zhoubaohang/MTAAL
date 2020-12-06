import os
import copy
import json
import numpy as np
from utils import BASEPATH, DATASET
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tqdm import trange

class DataLoader(object):

    def __init__(self, dataset, mask_zero=True, split_mode={'label':1., 'unlabel':0.}):

        self.mask_zero = mask_zero
        
        self.split_mode = split_mode

        assert DATASET.get(dataset), "Please check the dataset name!"

        # handle the raw data files
        self.__get_vocabs(f"{BASEPATH}{DATASET[dataset]['vocab']}")
        
        train_cache = self.__get_sentences(f"{BASEPATH}{DATASET[dataset]['to']['train']}")

        dev_cache = self.__get_sentences(f"{BASEPATH}{DATASET[dataset]['to']['dev']}")

        test_cache = self.__get_sentences(f"{BASEPATH}{DATASET[dataset]['to']['test']}")
        
        # calc the maximum length of the sentences
        self.max_sent_len = max([cache['max_sent_len'] for cache in [train_cache, dev_cache, test_cache]])
        self.max_char_len = max([cache['max_char_len'] for cache in [train_cache, dev_cache, test_cache]])
        
        # padding the sentences
        unlabel_cache = \
            self.__padding(train_cache['chars']+dev_cache['chars'],
                           train_cache['sentences']+dev_cache['sentences'],
                           train_cache['ner']+dev_cache['ner'],
                           train_cache['nen']+dev_cache['nen'])

        test_cache = \
            self.__padding(test_cache['chars'], test_cache['sentences'],
                           test_cache['ner'], test_cache['nen'])

        self.dataset = [
            np.array(unlabel_cache['real_len'].tolist()+test_cache['real_len'].tolist()),
            np.vstack((unlabel_cache['sentences'], test_cache['sentences'])),
            np.vstack((unlabel_cache['ner'], test_cache['ner'])),
            np.vstack((unlabel_cache['nen'], test_cache['nen'])),
            np.vstack((unlabel_cache['chars'], test_cache['chars']))
        ]

        # split the dataset
        self.__split()

        for k, v in self.split_idx.items():
            print(f"{k} nums: {len(v)}")

    @property
    def CharLength(self):
        return self.max_char_len

    @property
    def SentenceLength(self):
        return self.max_sent_len
    
    @property
    def CharSize(self):
        return max(self.chars.values()) + 1

    @property
    def VocabSize(self):
        return max(self.vocabs.values()) + 1

    @property
    def EmbedSize(self):
        return 200
    
    @property
    def NERTagSize(self):
        return len(self.ner_tags)
    
    @property
    def NENTagSize(self):
        return len(self.nen_tags)
    
    def idx2sentence(self, idx):
        sentences = []
        
        for i in idx:
            tmp = []
            for w in i:
                if w:
                    tmp.append(\
                        list(self.vocabs.keys())[\
                            list(self.vocabs.values()).index(int(w))])
            sentences.append(tmp)
        return sentences

    def idx2char(self, idx):
        sentences = []

        for i in idx:
            tmp = []
            for c in i:
                tmp.append(\
                    list(self.chars.keys())[\
                        list(self.chars.values()).index(int(c))])
            sentences.append(tmp)
        return sentences

    def idx2tag(self, idx, dtype='ner'):
        data = []
        tags = self.ner_tags if dtype == 'ner' else self.nen_tags

        for i in idx:
            tmp = []
            for w in i:
                if w == -1:
                    tmp.append('O')
                else:
                    tmp.append(tags[w])
            data.append(tmp)
        return np.array(data)

    def changeData(self, filtered_idx, selected_idx):
        unlabel_idx = self.split_idx['unlabel']
        label_idx = self.split_idx['label']
        self.split_idx["unlabel"] = filtered_idx
        self.split_idx["label"] = np.array(label_idx.tolist()\
                                          +selected_idx.tolist())

    def __mask_data(self, data, mask_rate):
        prob = np.random.uniform(size=data.shape)
        mask = np.ones_like(data)
        idx = np.where(prob <= mask_rate)
        mask[idx] = 0.

        data = data * mask + (1. - mask) * self.vocabs.get("unk")
        return data
    
    def getBatch(self, dtype, batch_size):
        assert batch_size > 0, f"batch_size {batch_size} <= 0, error!"
        idx = self.split_idx[dtype]
        np.random.shuffle(idx)

        sliced_idx = idx[:batch_size]
        data = [x[sliced_idx] for x in self.dataset]

        return (sliced_idx, data[0], data[1], data[2], data[3], data[4])

    def parseId2RawData(self, idx):
        data = [x[idx] for x in self.dataset]
        real_len = data[0]
        sent = self.idx2sentence(data[1])
        tag = self.idx2tag(data[2])
        raw_sent, raw_tag = [], []

        for i in range(len(data[0])):
            raw_sent.append(sent[i][-real_len[i]:])
            raw_tag.append(tag[i][-real_len[i]:])
        
        return (raw_sent, raw_tag)
    
    def getData(self, dtype, batch_size=-1):

        idx = self.split_idx[dtype]
        np.random.shuffle(idx)

        batch_size = len(idx) if batch_size < 0 else batch_size

        for i in trange(len(idx) // batch_size, ascii=True):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            sliced_idx = idx[start_idx:end_idx]
            data = [x[sliced_idx] for x in self.dataset]

            yield (sliced_idx, data[0], data[1], data[2], data[3], data[4])
    
    def nextBatch(self, batch_size, dtype, mask_rate=0.):

        idx = self.split_idx[dtype]
        np.random.shuffle(idx)

        for i in trange(len(idx) // batch_size, ascii=True):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            data = [x[idx[start_idx:end_idx]] for x in self.dataset]

            yield (data[0], self.__mask_data(data[1], mask_rate), data[2], data[3], data[4])
    
    def __split(self):
        idx = [i for i in range(len(self.dataset[0]))]
        np.random.shuffle(idx)

        start_idx = 0
        split_idx = {}
        for k, v in self.split_mode.items():
            end_idx = int(start_idx + len(idx)*v)
            split_idx[k] = np.array(idx[start_idx:end_idx])
            start_idx = end_idx

        self.split_idx = split_idx

    def __padding(self, chars, sentences, ner_tags, nen_tags):
        real_len = np.array([len(s) for s in sentences])
        x = pad_sequences(sentences, self.max_sent_len, value=int(not self.mask_zero))
        ner_y = pad_sequences(ner_tags, self.max_sent_len, value=-1)
        nen_y = pad_sequences(nen_tags, self.max_sent_len, value=-1)

        c = []
        for char in chars:
            tmp = []
            for t in char:
                pad_size = self.max_char_len - len(t)
                t = [self.chars.get('pad')]*(pad_size//2) + t
                pad_size = self.max_char_len - len(t)
                if pad_size:
                    t = t + [self.chars.get('pad')]*pad_size
                tmp.append(t)
            for _ in range(self.max_sent_len - len(tmp)):
                tmp.insert(0, [0]*self.max_char_len)
            c.append(tmp)

        return {
            'real_len': real_len,
            'sentences': x,
            'chars': c,
            'ner': ner_y,
            'nen': nen_y
        }

    def __get_vocabs(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

            self.chars = dict((c, i+1) for i, c in enumerate(data['chars']))
            self.vocabs = dict((w.lower(), i+1 if self.mask_zero else i) for i, w in enumerate(data['vocabs']))
            self.embeddings = np.array(data['w2v'])
            self.embedding_size = data['embedding_size']
            self.ner_tags = data['ner_tag']

            # Remove 'O' Tag
            self.nen_tags = data['nen_tag']
            if '-1' in self.nen_tags:
                del(self.nen_tags[self.nen_tags.index('-1')])
    
    def __get_sentences(self, path):
        max_sent_len = 0
        max_char_len = 0
        sentences = []
        chars = []
        ner_tags = []
        nen_tags = []
        with open(path, 'r', encoding='utf-8') as fp:
            char = []
            sentence = []
            ner_tag = []
            nen_tag = []
            for line in fp.readlines():
                line = line.strip()

                if line:
                    word, r_tag, n_tag = line.split('\t')
                    char.append([self.chars.get(i, self.chars.get('unk')) for i in word.lower()])
                    max_char_len = max(max_char_len, len(char[-1]))
                    sentence.append(self.vocabs.get(word.lower(), self.vocabs.get("unk")))
                    ner_tag.append(self.ner_tags.index(r_tag))
                    if n_tag == '-1':
                        nen_tag.append(self.nen_tags.index('O'))
                    else:
                        if '|' in n_tag:
                            n_tag = n_tag.split('|')[0]
                        nen_tag.append(self.nen_tags.index(n_tag))
                else:
                    if len(sentence) > 5 and len(sentence) < 60:
                        flag = False
                        # select the sentence sample without all "O" tags
                        for n, r in zip(nen_tag, ner_tag):
                            if n != 'O' and r != 'O':
                                flag = True
                        # flag = True

                        if flag:
                            chars.append(copy.deepcopy(char))
                            max_sent_len = max(max_sent_len, len(sentence))
                            sentences.append(copy.deepcopy(sentence))
                            ner_tags.append(copy.deepcopy(ner_tag))
                            nen_tags.append(copy.deepcopy(nen_tag))

                    char.clear()
                    sentence.clear()
                    ner_tag.clear()
                    nen_tag.clear()
                    
        return {
            'max_sent_len': max_sent_len,
            'max_char_len': max_char_len,
            'sentences': sentences,
            'chars': chars,
            'ner': ner_tags,
            'nen': nen_tags
        }

if __name__ == '__main__':

    dataloader = DataLoader('cdr')
    # print(dataloader.vocabs)
    # for real_len, x, ner, nen, c in dataloader.nextBatch(64, 'label'):
    #     print(dataloader.idx2sentence(x))