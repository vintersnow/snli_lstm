from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np

rnn = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}


class SNLIRNN(nn.Module):
    '''直列型モデル'''

    def __init__(self, vocab, hps, rnn_type='lstm', acti=nn.LogSoftmax(1)):
        super(SNLIRNN, self).__init__()
        self.vocab = vocab
        self.hps = hps
        self.rnn_type = rnn_type

        self.embd = nn.Embedding(
            vocab.size, hps.embd_size, padding_idx=vocab.pad_id)

        if hps.pre_embd != '':
            embd_w = np.load(hps.pre_embd)
            assert vocab.size == embd_w.shape[0]
            assert hps.embd_size == embd_w.shape[1]
            self.embd.weight.data = torch.from_numpy(embd_w.astype(np.float32))
            self.embd.weight.requires_grad = not hps.no_train_embd

        self.rnn1 = rnn[rnn_type](
            hps.embd_size,
            hps.hidden_size,
            hps.num_layers,
            batch_first=True,
            dropout=hps.dropout,
            bidirectional=hps.bidirectional)

        self.rnn2 = rnn[rnn_type](
            hps.embd_size,
            hps.hidden_size,
            hps.num_layers,
            batch_first=True,
            dropout=hps.dropout,
            bidirectional=hps.bidirectional)

        hsize = hps.hidden_size * (2 if hps.bidirectional else 1)
        self.ouput_ly = nn.Linear(hsize, 3)
        self.activation = acti

    def forward(self, s1, s1_len, s1_mask, s2, s2_len):
        embd1 = self.embd(s1)
        embd2 = self.embd(s2)

        s1_len_n = s1_len.cpu().data.numpy()
        # padding部分を無視するためにpackする
        packed = pack_padded_sequence(embd1, s1_len_n, batch_first=True)
        output, hidden = self.rnn1(packed)

        # output (B*L*U)
        output, _ = self.rnn2(embd2, hidden)

        idx = (s2_len - 1).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand_as(output)  # (B*L*U)
        output = torch.gather(output, 1, idx)
        output = output[:, 0]

        output = self.ouput_ly(output)

        if not self.activation:
            return output
        return self.activation(output)


class HypoModel(nn.Module):
    def __init__(self, vocab, hps, rnn_type='lstm', acti=nn.LogSoftmax(1)):
        super(HypoModel, self).__init__()
        self.vocab = vocab
        self.hps = hps
        self.rnn_type = rnn_type

        self.embd = nn.Embedding(
            vocab.size, hps.embd_size, padding_idx=vocab.pad_id)

        if hps.pre_embd != '':
            embd_w = np.load(hps.pre_embd)
            assert vocab.size == embd_w.shape[0]
            assert hps.embd_size == embd_w.shape[1]
            self.embd.weight.data = torch.from_numpy(embd_w.astype(np.float32))
            self.embd.weight.requires_grad = not hps.no_train_embd

        self.rnn = rnn[rnn_type](
            hps.embd_size,
            hps.hidden_size,
            hps.num_layers,
            batch_first=True,
            dropout=hps.dropout,
            bidirectional=hps.bidirectional)

        hsize = hps.hidden_size * (2 if hps.bidirectional else 1)
        # self.ouput_ly = nn.Linear(hsize, 4)
        self.ouput_ly = nn.Linear(hsize, 3)
        self.activation = acti

    def forward(self, s1, s1_len, s1_mask, s2, s2_len):
        embd2 = self.embd(s2)

        # output (B*L*U)
        output, _ = self.rnn(embd2)

        idx = (s2_len - 1).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand_as(output)  # (B*L*U)
        output = torch.gather(output, 1, idx)
        output = output[:, 0]

        output = self.ouput_ly(output)

        if not self.activation:
            return output
        return self.activation(output)
