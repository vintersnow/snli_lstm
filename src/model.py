from torch import nn


class SNLIRNN(nn.Module):
    def __init__(self, vocab, hps, rnn_type='lstm'):
        super(SNLIRNN, self).__init__()
        self.vocab = vocab
        self.hps = hps
        self.rnn_type = rnn_type
