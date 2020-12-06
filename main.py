import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from params import params
from keras import backend as K
from models import AMT4NERaNEN
from data_loader import DataLoader
from activelearning import ActiveLearning as AL
from utils import save_result, calc_ner_f1, calc_nen_f1
tf.compat.v1.reset_default_graph()

np.random.seed(0)
tf.compat.v1.set_random_seed(0)
# init dataloader
dataloader = DataLoader(params.dataset,
                        mask_zero=params.encoder_type=='lstm',
                        split_mode=params.split_mode)
params.embeddings = dataloader.embeddings
params.embeddings_size = dataloader.embedding_size

# Model's initial parameters
BATCH_SIZE = params.batch_size
EMBED_LEN = dataloader.EmbedSize
CHAR_LEN = dataloader.CharLength
VOCABS_SIZE = dataloader.VocabSize
CHARCAB_SIZE = dataloader.CharSize
NER_TAG_SIZE = dataloader.NERTagSize
NEN_TAG_SIZE = dataloader.NENTagSize
SENTENCE_LEN = dataloader.SentenceLength

print(f"VOCAB SIZE {VOCABS_SIZE}")

# init model
model = AMT4NERaNEN(CHAR_LEN, CHARCAB_SIZE, SENTENCE_LEN, VOCABS_SIZE,
                    EMBED_LEN, NER_TAG_SIZE, NEN_TAG_SIZE, params)
tensor_params = model()
al = AL(params.al)

# result cache
NER_F1 = []
NEN_F1 = []

# init Tensorflow Session
if params.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    gpuConfig = tf.ConfigProto(allow_soft_placement=True)
    gpuConfig.gpu_options.allow_growth = True
    sess = tf.Session(config=gpuConfig)
else:
    sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())

# Begin to train
for epc in range(params.epoch):

    # Active Learning
    if al.al_type == 'diversity':
        unlabel_idx = []
        real_len = []
        probs = []

        for _ in range(1):
            _, _, x, _, _, c = dataloader.getBatch('label', params.batch_size)
            _, _, u_x, _, _, u_c = dataloader.getBatch('unlabel', params.batch_size)
            feed_dict = {
                tensor_params['c']: c,
                tensor_params['u_c']: u_c,
                tensor_params['x']: x,
                tensor_params['u_x']: u_x
            }
            # Diversity Training
            dis_en_loss, _, dis_de_loss, _ = sess.run([
                tensor_params['discrim_encoder_loss'],
                tensor_params['discrim_encoder_solver'],
                tensor_params['discrim_decoder_loss'],
                tensor_params['discrim_decoder_solver']
            ], feed_dict=feed_dict)

            tqdm.write(f"Div D:{dis_de_loss} G:{dis_en_loss}")

        for i, l, x, _, _, c in dataloader.getData('unlabel', batch_size=params.batch_size):
            feed_dict = { tensor_params['x']: x, tensor_params['c']: c }
            prob = sess.run(tensor_params['dis_label'], feed_dict=feed_dict)
            unlabel_idx += i.tolist()
            real_len += l.tolist()
            probs.append(prob)

        unlabel_idx = np.array(unlabel_idx)
        real_len = np.array(real_len)
        probs = np.vstack(probs)
        sorted_idx = al.diversity(probs, real_len, params.query_num)
        selected_idx = unlabel_idx[sorted_idx]
        filtered_idx = np.delete(unlabel_idx, sorted_idx)
        dataloader.changeData(filtered_idx, selected_idx)
    else:
        unlabel_idx = []
        real_len = []
        ner_prob = []
        nen_prob = []

        for i, l, x, _, _, c in dataloader.getData('unlabel', batch_size=params.batch_size):
            feed_dict = { tensor_params['x']: x, tensor_params['c']: c }
            n_p, r_p = sess.run([tensor_params['nen_prob'],
            		     tensor_params['ner_prob']], feed_dict=feed_dict)
            unlabel_idx += i.tolist()
            real_len += l.tolist()
            ner_prob.append(r_p)
            nen_prob.append(n_p)
        
        unlabel_idx = np.array(unlabel_idx)
        real_len = np.array(real_len)
        ner_prob = np.vstack(ner_prob)
        nen_prob = np.vstack(nen_prob)
        sorted_idx = al(ner_prob, nen_prob, real_len, params.query_num)
        selected_idx = unlabel_idx[sorted_idx]
        filtered_idx = np.delete(unlabel_idx, sorted_idx)
        dataloader.changeData(filtered_idx, selected_idx)

    # Multi-task Learning

    # Task Discriminator Training
    if params.ad_task:
        for real_len, x, ner, nen, c in dataloader.nextBatch(params.batch_size, 'label'):
            feed_dict = {
                tensor_params['c']: c,
                tensor_params['x']: x
            }

            discrim_private_loss, _ = sess.run([tensor_params['discrim_private_loss'],
                                                tensor_params['discrim_private_solver']],
                                                feed_dict=feed_dict)
            discrim_share_loss, _ = sess.run([tensor_params['discrim_share_loss'],
                                                tensor_params['discrim_share_solver']],
                                                feed_dict=feed_dict)

            _, _, x, _, _, c = dataloader.getBatch('unlabel', params.batch_size)
            feed_dict = {
                tensor_params['c']: c,
                tensor_params['x']: x
            }

            discrim_private_loss, _ = sess.run([tensor_params['discrim_private_loss'],
                                                tensor_params['discrim_private_solver']],
                                                feed_dict=feed_dict)
            discrim_share_loss, _ = sess.run([tensor_params['discrim_share_loss'],
                                                tensor_params['discrim_share_solver']],
                                                feed_dict=feed_dict)
 
    # Task Training
    for _ in range(1):
        for real_len, x, ner, nen, c in dataloader.nextBatch(params.batch_size, 'label'):
            feed_dict = {
                tensor_params['c']: c,
                tensor_params['x']: x,
                tensor_params['ner_y']: ner,
                tensor_params['nen_y']: nen
            }
            nen_prob, ner_prob, loss, _ = sess.run([tensor_params['nen_prob'], tensor_params['ner_prob'],
                                                    tensor_params['loss'], tensor_params['solver']],
                                                    feed_dict=feed_dict)
            ner_f1 = calc_ner_f1(dataloader.idx2tag(ner), 
                                dataloader.idx2tag(np.argmax(ner_prob, axis=-1)), real_len, tags=dataloader.ner_tags)[2]
            nen_f1 = calc_nen_f1(dataloader.idx2tag(nen, dtype='nen'), 
                                dataloader.idx2tag(np.argmax(nen_prob, axis=-1), dtype='nen'), real_len)[3]

            # Print Result
            # tqdm.write(f"loss:{loss} F1 NER:{ner_f1} NEN:{nen_f1}")

    # Test
    ner_f1s, nen_f1s = [], []
    for _, real_len, x, ner, nen, c in dataloader.getData('test', batch_size=params.batch_size):
        feed_dict = {
            tensor_params['x']: x,
            tensor_params['c']: c
        }
        nen_prob, ner_prob = sess.run([tensor_params['nen_prob'],
                                        tensor_params['ner_prob']], feed_dict=feed_dict)
        ner_f1 = calc_ner_f1(dataloader.idx2tag(ner), 
                                dataloader.idx2tag(np.argmax(ner_prob, axis=-1)), real_len, tags=dataloader.ner_tags)[2]
        nen_f1 = calc_nen_f1(dataloader.idx2tag(nen, dtype='nen'), 
                                dataloader.idx2tag(np.argmax(nen_prob, axis=-1), dtype='nen'), real_len)[3]
        ner_f1s.append(ner_f1)
        nen_f1s.append(nen_f1)
    ner_f1 = np.mean(ner_f1s)
    nen_f1 = np.mean(nen_f1s)
    print(f"test >>>>>>>>>>> NER:{ner_f1} NEN:{nen_f1}")
    NER_F1.append(ner_f1)
    NEN_F1.append(nen_f1)

sess.close()

model_name = f"MTAAL-{params.al}" if params.ad_task else f"MTAL-{params.al}"
save_result('results/result.json', params.dataset, model_name, NER_F1, NEN_F1)