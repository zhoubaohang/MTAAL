import numpy as np

class ActiveLearning(object):

    def __init__(self, method):
        assert method in ['lc', 'mnlp', 'random', 'entropy', 'diversity'], "active learning method is not found"
        self.al_type = method

    def __call__(self, ner_prob, nen_prob, real_len, query_num):

        func = getattr(self, self.al_type)
        return func(ner_prob, nen_prob, real_len, query_num)
    
    def diversity(self, probs, real_len, query_num):
        score = []

        # max_prob = np.max(probs, axis=-1)
        max_prob = probs[:, :, 1]

        for i, l in enumerate(real_len):
            score.append(1. - np.prod(max_prob[i, -l:]))
        
        return np.argsort(score)[-query_num:]

    def entropy(self, ner_prob, nen_prob, real_len, query_num):
        ner_score = []
        nen_score = []

        enp_ner_prob = np.sum(- ner_prob * np.log(ner_prob), axis=-1)
        enp_nen_prob = np.sum(- nen_prob * np.log(nen_prob), axis=-1)

        for i, l in enumerate(real_len):
            ner_score.append(np.max(enp_ner_prob[i, -l:]))
            nen_score.append(np.max(enp_nen_prob[i, -l:]))

        score = np.array(ner_score) + np.array(nen_score)

        return np.argsort(score)[-query_num:]

    def lc(self, ner_prob, nen_prob, real_len, query_num):
        ner_score = []
        nen_score = []

        max_ner_prob = np.max(ner_prob, axis=-1)
        max_nen_prob = np.max(nen_prob, axis=-1)

        for i, l in enumerate(real_len):
            ner_score.append(1. - np.prod(max_ner_prob[i, -l:]))
            nen_score.append(1. - np.prod(max_nen_prob[i, -l:]))
        
        score = np.array(ner_score) + np.array(nen_score)

        return np.argsort(score)[-query_num:]
    
    def mnlp(self, ner_prob, nen_prob, real_len, query_num):
        ner_score = []
        nen_score = []

        max_ner_prob = np.max(ner_prob, axis=-1)
        max_nen_prob = np.max(nen_prob, axis=-1)

        for i, l in enumerate(real_len):
            ner_score.append(- np.log(np.mean(max_ner_prob[i, -l:])))
            nen_score.append(- np.log(np.mean(max_nen_prob[i, -l:])))

        score = np.array(ner_score) + np.array(nen_score)

        return np.argsort(score)[-query_num:]

    def random(self, ner_prob, nen_prob, real_len, query_num):
        ner_score = []
        nen_score = []

        max_ner_prob = np.max(ner_prob, axis=-1)
        max_nen_prob = np.max(nen_prob, axis=-1)

        for i, l in enumerate(real_len):
            ner_score.append(1. - np.prod(max_ner_prob[i, -l:]))
            nen_score.append(1. - np.prod(max_nen_prob[i, -l:]))
        
        score = np.array(ner_score) + np.array(nen_score)

        return np.random.choice(np.argsort(score), query_num)