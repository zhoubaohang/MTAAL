# MTAAL
[Title] MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization

[Authors] Baohang Zhou, Xiangrui Cai, Ying Zhang, Wenya Guo, Xiaojie Yuan

[AAAI 2021 paper (Waiting for publication)]() [[video](https://slideslive.com/38949282/mtaal-multitask-adversarial-active-learning-for-medical-named-entity-recognition-and-normalization?ref=account-folder-75497-folders)]

## Preparation
1. Clone the repo to your local.
2. Download Python version: 3.6.5.
3. Download the word embeddings from the following websites. Put them into the "pretrain" folder. ([Word2Vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and [Glove]())
4. Open the shell or cmd in this repo folder. Run this command to install necessary packages.
```cmd
pip install -r requirements.txt
```

## Experiments
1. Before running models, you should run this command to handle the dataset. You can choose the parameters to change datasets and word embeddings.
```cmd
python preprocess.py --dataset=[ncbi, cdr] --wordembedding=[word2vec, glove]
```
2. You can input the following command to run the different active learning models. There are different choices for some parameters shown in []. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
| epoch | int | Query times for active learning |
| label | float | The split proportion for initial labeled set |
| unlabel | float | The split proportion for initial unlabeled set |
| test | float | The split proportion for test set |
| query_num | int | The number of query samples |
| ad_task | str | Choose whether to use Task Adversarial Learning |
| task | str | Choose the task to run model. "all" is multi-task scenario |
| al | str | Choose the active learning method. |

```cmd
python main.py params \
--epoch=70 \
--label=0.2 \
--unlabel=0.7 \
--test=0.1 \
--batch_size=32 \
--query_num=64 \
--ad_task=[True, False] \
--dataset=[ncbi, cdr] \
--rnn_units=64 \
--task=[all, ner, nen] \
--gpu=[True, False] \
--al=[diversity, random, lc, entropy, mnlp]
```
3. After running the model, the test result is saved in the "results" folder.

PS: We use the evaluation metrics as described in this paper ([Zhao et al.](https://doi.org/10.1609/aaai.v33i01.3301817)).