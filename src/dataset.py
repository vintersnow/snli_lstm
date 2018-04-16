from torch.utils.data import Dataset, DataLoader
import torch
from torchutils.data.transformer import Text2Id, ToTensor, ClipText
from torchutils.data import merge_fn
import json


class SNLIDataset(Dataset):
    def __init__(self, file, transform=None):
        self.file = file
        self.transform = transform
        with open(file, encoding='utf-8') as f:
            self.data = json.load(f)
        self.transformed = [None] * len(self.data)
        self.labels = {
            'neutral': 0,
            'entailment': 1,
            'contradiction': 2,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transformed[idx]:
            return self.transformed[idx]

        sample = self.data[idx]
        sample['label'] = self.labels[sample['label']]

        if self.transform:
            if isinstance(self.transform, list):
                for c in self.transform:
                    sample = c(sample)
            else:
                sample = self.transform(sample)

        self.transformed[idx] = sample

        return sample


def make_dataloader(data_file, batch_size, vocab, max_len, single_pass):

    transforms = [
        ClipText(max_len, 's1', 's2'),
        Text2Id(vocab, 's1', 's2'),
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
        num_workers=1,
        collate_fn=collate_fn)

    return dataloader


def next_batch(dataloader):
    while True:
        for batch in dataloader:
            yield batch
