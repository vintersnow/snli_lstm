from dataset import make_dataloader
from torchutils.data import Vocab


def dataset():
    vocab = Vocab('data/vocab', 50000)
    dataloader = make_dataloader('data/conv_dev.jsonl', 32, vocab, 200, False,
                                 False)
    for batch in dataloader:
        for key in batch:
            if key != 'id':
                print(key, batch[key].size())
        break


if __name__ == '__main__':
    dataset()

