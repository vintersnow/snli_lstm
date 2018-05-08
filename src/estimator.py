from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from dataset import make_dataloader
import torch.nn.functional as F
import json
import torch
from torchutils import get_logger, get_vars
import numpy as np

logger = get_logger(__name__)


class RNNEstimator(BaseEstimator):
    def __init__(self, model, vocab, max_len, use_cuda=False, activate=True):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        self.use_cuda = use_cuda
        self.activate = activate

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        '''
        x (list(sample(dict)))
            sample (dict): keys are 's1', 's2', 'label'
        '''
        dataloader = make_dataloader(x, 64, self.vocab, self.max_len, False,
                                     self.use_cuda)
        preds = []
        for batch in dataloader:
            keys = ('s1', 's1_len', 's1_mask', 's2', 's2_len')
            outputs = self.model(
                *get_vars(batch, *keys, use_cuda=self.use_cuda))  # (B*3)
            if self.activate:
                outputs = F.softmax(outputs, 1)
            else:
                outputs = F.log_softmax(outputs, 1)
            preds.extend(outputs.cpu().data.tolist())
        return np.asarray(preds)

    def predict(self, x):
        preds = self.predict_proba(x)
        preds = np.argmax(preds, 1)
        return preds

    def score(self, x, y):
        preds = self.predict(x)
        score = f1_score(y, preds, average='macro')
        return score


class Preprocess(BaseEstimator):
    def __init__(self):
        super(Preprocess, self).__init__()

    def fit(self, x, y):
        return self

    def transform(self, x):
        return self.make_dict([self.split_raw_text(xx) for xx in x])

    def make_raw_text(self, data):
        '''
        data (dict)
            s1 (list(str))
            s2 (list(str))
        '''
        s1 = data['s1']
        s2 = data['s2']
        return ' '.join([s + '_1' for s in s1] + [s + '_2' for s in s2])

    def split_raw_text(self, x):
        '''
        x: s1 + s2
        '''
        # print(x)
        arr = x.split(' ')
        s1 = [s for s in arr if '_1' in s]
        s1 = [s.replace('_1', '') for s in s1]
        s2 = [s for s in arr if '_2' in s]
        s2 = [s.replace('_2', '') for s in s2]
        return s1, s2

    def make_dict(self, x):
        return [{'s1': s[0], 's2': s[1], 'label': 'neutral'} for s in x]


def make_test_data(data_path):
    logger.info('loading text...')
    with open(data_path, encoding='utf-8') as f:
        test_data = json.load(f)
    logger.info('data size: %s' % len(test_data))

    return test_data


if __name__ == '__main__':
    from torchutils.data import Vocab
    from torch.autograd import Variable
    from sklearn.pipeline import make_pipeline

    data = make_test_data('./data/small_test.json')
    vocab = Vocab('./data/vocab', 0)

    class Dummay(object):
        def __call__(self, *args):
            return Variable(torch.FloatTensor([[0, 0, 1]]))

    model = Dummay()
    rnnest = RNNEstimator(model, vocab, 50)
    pre = Preprocess()
    pipe = make_pipeline(pre, rnnest)

    raw_data = pre.make_raw_text(data[0])
    # print(raw_data)
    print(pipe.predict_proba([raw_data]))

