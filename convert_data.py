import json
from nltk.tokenize import word_tokenize
from collections import defaultdict
from argparse import ArgumentParser
from os import path, mkdir

parser = ArgumentParser('python convert_data.py')
parser.add_argument('--data_path', type=str, default='data',
                    help='path to vocabulary file')
dpath = parser.parse_args().data_path
out_dir = path.join(dpath, 'convert')

vocab = {}


def read(file):
    data = []

    def tokenize(s):
        return [w.lower() for w in word_tokenize(s)]

    with open(file) as f:
        for line in f:
            j = json.loads(line)
            if j['gold_label'] == '-':
                print('skip')
                continue
            ex = {}
            ex['label'] = j['gold_label']
            ex['s1'] = tokenize(j['sentence1'])
            ex['s2'] = tokenize(j['sentence2'])
            ex['id'] = j['pairID']
            data.append(ex)
    return data


def write_out(data, out_file):
    with open(path.join(out_dir, out_file), 'w') as f:
        json.dump(data, f)


def make_vocab(data):
    vocab = defaultdict(int)
    for ex in data:
        for w in ex['s1']:
            vocab[w] += 1
        for w in ex['s2']:
            vocab[w] += 1
    return vocab


def main():
    dev_data = read(path.join(dpath, 'snli_1.0_dev.jsonl'))
    test_data = read(path.join(dpath, 'snli_1.0_test.jsonl'))
    train_data = read(path.join(dpath, 'snli_1.0_train.jsonl'))

    if not path.isdir(out_dir):
        mkdir(out_dir)
    write_out(dev_data, 'conv2_dev.json')
    write_out(test_data, 'conv2_test.json')
    write_out(train_data, 'conv2_train.json')

    vocab = make_vocab(train_data)

    with open(path.join(dpath, 'vocab'), 'w') as f:
        for w, v in sorted(vocab.items(), key=lambda x: -x[1]):
            f.write('%s %d\n' % (w, v))


if __name__ == '__main__':
    main()
