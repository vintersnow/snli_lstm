from torch.utils.data import Dataset, DataLoader
import torch
from torchutils.data.transformer import Text2Id, ToTensor, ClipText
from torchutils.data import merge_fn, Vocab
import json
from os import path
import tempfile

unk_hypo = False


class SNLIDataset(Dataset):
    labels = {
        'neutral': 0,
        'entailment': 1,
        'contradiction': 2,
    }

    def __init__(self, data, transform=None):
        if type(data) is list:
            self.data = data
        elif type(data) is str:
            assert path.isfile(data)
            self.file = data
            with open(data, encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            raise ValueError('data should be dict or file: %s.' % type(data))
        self.transform = transform
        self.transformed = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transformed[idx]:
            return self.transformed[idx]

        sample = self.data[idx]
        sample['label'] = SNLIDataset.labels[sample['label']]

        if self.transform:
            if isinstance(self.transform, list):
                for c in self.transform:
                    sample = c(sample)
            else:
                sample = self.transform(sample)

        self.transformed[idx] = sample

        return sample


class Guard(object):
    def __init__(self, uid, *keys):
        self.uid = uid
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            if len(sample[key]) == 0:
                sample[key] += [self.uid]
                sample[key + '_len'] = 1

        return sample


def make_dataloader(data_file, batch_size, vocab, max_len, single_pass):

    if unk_hypo:
        unk_file = tempfile.NamedTemporaryFile('w')
        unk_vocab = Vocab(unk_file.name, 0)
        t2i = (Text2Id(unk_vocab, 's1'), Text2Id(vocab, 's2'))
    else:
        t2i = (Text2Id(vocab, 's1', 's2'),)

    transforms = [
        ClipText(max_len, 's1', 's2'),
        *t2i,
        Guard(vocab.unk_id, 's1', 's2'),
        ToTensor({
            's1': torch.LongTensor,
            's2': torch.LongTensor
        })
    ]
    dataset = SNLIDataset(data_file, transforms)

    collate_fn = merge_fn(['s1', 's2'], ['id'], lambda x: -x['s1_len'])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not single_pass,
        num_workers=0,
        collate_fn=collate_fn)

    return dataloader


def next_batch(dataloader):
    while True:
        for batch in dataloader:
            yield batch
