import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Input, Sequential
from keras.losses import categorical_crossentropy
from utils import categorical_cross_entropy as CCE
from keras_self_attention import SeqSelfAttention as Attention
from keras.layers import (
    LSTM,
    Dense,
    Conv1D,
    Lambda,
    Dropout,
    MaxPool1D,
    GlobalMaxPool1D,
    Embedding,
    Bidirectional,
    TimeDistributed
)

one_hot = lambda y, len_tag: K.one_hot(K.cast(y, tf.int32), len_tag)

class AMT4NERaNEN(object):
    '''
        Attention Based Multi-Task Learning for NER and NEN
    '''

    def __init__(self, len_char, len_charcab, len_sent, len_vocab, len_emd, len_ner_tag, len_nen_tag, 
                  params):
        # Word Embedding Size
        self.len_word_emd = len_emd
        # Char Embedding Size
        self.len_char_emd = 30
        # Maximum Char Length
        self.len_char = len_char
        # Maximum Sentence Length
        self.len_sent = len_sent
        # Vocabulary Size
        self.len_vocab = len_vocab
        # Char Vocabulary Size
        self.len_charcab = len_charcab
        # NER Tag Size
        self.len_ner_tag = len_ner_tag
        # NEN Tag Size
        self.len_nen_tag = len_nen_tag
        # Task Type
        self.task = params.task

        # Hyper-parameters
        self.lr = params.lr
        self.beta1 = params.beta1
        self.beta2 = params.beta2
        self.rnn_units = params.rnn_units
        self.encoder_type = params.encoder_type
        
        # Pretrained Word Embeddings config
        self.len_word_emd = params.embeddings_size
        self.embeddings = params.embeddings
        if self.encoder_type=="lstm":
            self.embeddings = np.vstack((np.array([0.]*self.len_word_emd),
                                         self.embeddings))
    
    def __init_layers(self, keep_prob=0.):
        # Embedding Layer
        self.char_emd = Embedding(input_dim=self.len_charcab,
                                  output_dim=self.len_char_emd,
                                  mask_zero=True,
                                  trainable=True)
        self.word_emd = Embedding(input_dim=self.len_vocab,
                                  output_dim=self.len_word_emd,
                                  weights=[self.embeddings],
                                  mask_zero=True if self.encoder_type=="lstm" else False,
                                  trainable=False)
        
        # LSTM Encoder Layer
        self._lstm_encoder = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=True))
        ])

        # CNN Layer
        self.conv1 = Conv1D(self.rnn_units, 3, border_mode='same')
        self.conv2 = Conv1D(self.rnn_units, 3, border_mode='same')

        # NER Decoder Layer
        self._ner_decoder = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=True)),
            TimeDistributed(Dense(self.len_ner_tag, activation=K.softmax))
        ])

        # NEN Decoder Layer
        self._nen_decoder = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=True)),
            TimeDistributed(Dense(self.len_nen_tag, activation=K.softmax))
        ])

        # Task Discriminator Layer
        self.discrim_task = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=True)),
            TimeDistributed(Dense(2, activation=K.softmax))
        ])

        # Diversity Discriminator Layer
        self.discrim_diversity = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=True)),
            TimeDistributed(Dense(2, activation=K.softmax))
        ])
    
    
    def ner_decoder(self, x):
        x = Attention(attention_activation='tanh')(x)
        return x, self._ner_decoder(x)

    def nen_decoder(self, x):
        x = Attention(attention_activation='tanh')(x)
        return x, self._nen_decoder(x)

    def _cnn_encoder(self, inputs):
        '''
            1D Convolution Neural Networks
        '''
        conv1_output = self.conv1(inputs)
        conv2_output = self.conv2(conv1_output)
        outputs = K.concatenate([inputs, conv2_output], axis=-1)

        return outputs
    
    def _char_encoder(self, inputs, keep_prob=0.):
        '''
            Char Encoder Layer
        '''
        char_encoder = Sequential([
            Bidirectional(LSTM(units=self.rnn_units,
                               dropout=keep_prob,
                               return_sequences=False))
        ], name='char_encoder')
        outputs = char_encoder(inputs)
        return outputs

    @property
    def encoder(self):
        return getattr(self, f"_{self.encoder_type}_encoder")
    
    def __collect_vars(self):
        name_scopes = ['embedding', 'encoder', 'decoder', 'discrim_task', 'discrim_diversity']
        t_vars = tf.trainable_variables()
        cache = {}

        for name in name_scopes:
            tmp = [var for var in t_vars if name in var.name]
            cache[name] = tmp
        
        return cache
    
    def __call__(self):
        # Dropout rate
        keep_prob = 0.
        # init all layers parameters
        self.__init_layers(keep_prob)
        # char sequence placeholder
        c = Input(shape=(self.len_sent, self.len_char,))
        # char sequence of unlabel sentence
        u_c = Input(shape=(self.len_sent, self.len_char,))
        # sentence sequence placeholder
        x = Input(shape=(self.len_sent,))
        # sentence sequence of unlable sentence
        u_x = Input(shape=(self.len_sent,))
        # NER Task Tags
        ner_y = Input(shape=(self.len_sent,))
        ner_oh_y = one_hot(ner_y, self.len_ner_tag)
        # NEN Task Tags
        nen_y = Input(shape=(self.len_sent,))
        nen_oh_y = one_hot(nen_y, self.len_nen_tag)

        with tf.name_scope('embedding'):
            # handle labeled data
            word_emd = self.word_emd(x)
            char_emd = K.reshape(self.char_emd(c), [-1, self.len_char, self.len_char_emd])

            # handle unlabel data
            u_word_emd = self.word_emd(u_x)
            u_char_emd = K.reshape(self.char_emd(u_c), [-1, self.len_char, self.len_char_emd])
        
        with tf.name_scope('encoder'):
            # handle labeled data
            char_x = K.reshape(self._char_encoder(char_emd), [-1, self.len_sent, self.rnn_units*2])
            word_char_x = K.concatenate([word_emd, char_x], axis=-1)
            en_word_char_x = self.encoder(word_char_x)

            # handle unlabeled data
            u_char_x = K.reshape(self._char_encoder(u_char_emd), [-1, self.len_sent, self.rnn_units*2])
            u_word_char_x = K.concatenate([u_word_emd, u_char_x], axis=-1)
            u_en_word_char_x = self.encoder(u_word_char_x)
        
        with tf.name_scope('decoder'):
            ner_att, ner_prob = self.ner_decoder(en_word_char_x)
            nen_att, nen_prob = self.nen_decoder(en_word_char_x)

        with tf.name_scope('discrim_task'):
            dis_ner = self.discrim_task(ner_att)
            dis_nen = self.discrim_task(nen_att)

        with tf.name_scope('discrim_diversity'):
            dis_label = self.discrim_diversity(en_word_char_x)
            dis_unlabel = self.discrim_diversity(u_en_word_char_x)
        
        # Diversity Discriminator Loss
        discrim_decoder_loss = CCE(dis_label, tf.ones_like(dis_label)[:,:,0])+\
                               CCE(dis_unlabel, tf.zeros_like(dis_unlabel)[:,:,0])
        discrim_encoder_loss = CCE(dis_label, tf.ones_like(dis_label)[:,:,0])+\
                               CCE(dis_unlabel, tf.ones_like(dis_unlabel)[:,:,0])

        # Task Discriminator Loss
        discrim_private_loss = CCE(dis_ner, tf.ones_like(dis_ner)[:,:,0])+\
                               CCE(dis_nen, tf.zeros_like(dis_nen)[:,:,0])
        discrim_share_loss = - discrim_private_loss
            
        # NER Task Loss
        ner_loss = tf.reduce_mean(categorical_crossentropy(ner_oh_y, ner_prob))
        
        # NEN Task Loss
        nen_loss = tf.reduce_mean(categorical_crossentropy(nen_oh_y, nen_prob))

        # Trainable Parameters
        t_vars = self.__collect_vars()

        task_loss = {
            'all': nen_loss + ner_loss,
            'nen': nen_loss,
            'ner': ner_loss
        }
        assert self.task in task_loss.keys(), "task type is not in this model!"

        # Diversity Discriminator Solver
        discrim_decoder_solver = tf.compat.v1.train.AdamOptimizer(1e-4)\
                                             .minimize(discrim_decoder_loss, var_list=t_vars['discrim_diversity'])
        discrim_encoder_solver = tf.compat.v1.train.AdamOptimizer(1e-4)\
                                             .minimize(discrim_encoder_loss, var_list=t_vars['encoder'])

        # Task Solver
        loss = task_loss[self.task]
        solver = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2)\
                                   .minimize(loss)

        # Task Discriminator Solver
        discrim_private_solver = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.5)\
                                          .minimize(discrim_private_loss, var_list=t_vars['decoder']+t_vars['discrim_task'])
        discrim_share_solver = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.5)\
                                        .minimize(discrim_share_loss, var_list=t_vars['encoder'])
        
        return {
            'c': c,
            'x': x,
            'u_c': u_c,
            'u_x': u_x,
            'ner_y': ner_y,
            'nen_y': nen_y,
            'ner_prob': ner_prob,
            'nen_prob': nen_prob,
            'loss': loss,
            'solver': solver,
            'discrim_private_loss': discrim_private_loss,
            'discrim_private_solver': discrim_private_solver,
            'discrim_share_loss': discrim_share_loss,
            'discrim_share_solver': discrim_share_solver,
            'dis_label': dis_label,
            'discrim_encoder_loss': discrim_encoder_loss,
            'discrim_encoder_solver': discrim_encoder_solver,
            'discrim_decoder_loss': discrim_decoder_loss,
            'discrim_decoder_solver': discrim_decoder_solver
        }
