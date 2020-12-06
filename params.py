import fire

class Parameters(object):

    def params(self,
               dataset='ncbi',
               encoder_type="lstm",
               seed=0,
               lr=1e-3,
               label=.5,
               unlabel=0.,
               test=.5,
               epoch=10,
               beta1=0.9,
               beta2=0.9,
               gpu=False,
               task='all',
               al='random',
               ad_task=True,
               query_num=16,
               batch_size=32,
               rnn_units=64):

        self.lr = lr
        self.al = al
        self.gpu = gpu
        self.task = task
        self.seed = seed
        self.split_mode={
            'label': label,
            'unlabel': unlabel,
            'test': test
        }
        self.epoch = epoch
        self.beta1 = beta1
        self.beta2 = beta2
        self.ad_task = ad_task
        self.dataset = dataset
        self.rnn_units = rnn_units
        self.query_num = query_num
        self.batch_size = batch_size
        self.encoder_type = encoder_type

        self.embeddings = []
        self.embeddings_size = 0


params = Parameters()
fire.Fire(params)