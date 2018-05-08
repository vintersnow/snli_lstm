from model import SNLIRNN, HypoModel
from hyperparams import hps
from os import path
from task import train, test
from dataset import make_dataloader
from torchutils import Model
from torchutils.data import Vocab
from torchutils.misc import Timeit


def build_loader(vocab, hps):
    mode = hps.mode.replace('hypo', '')
    if mode == 'train':
        single_pass = False
        bsize = {'train': hps.batch_size, 'val': hps.batch_size}
    elif mode == 'val':
        single_pass = True
        bsize = {'val': hps.batch_size}
    elif mode == 'test':
        single_pass = True
        bsize = {'test': hps.batch_size}
    else:
        raise ValueError('Unknown mode: %s' % hps.mode)

    loader = {}
    args = (vocab, hps.max_steps, single_pass)
    for key in bsize:
        dpath = path.join(hps.data_path, getattr(hps, key + '_data'))
        loader[key] = make_dataloader(dpath, bsize[key], *args)

    return loader


def main(hps):
    timer = Timeit()

    with timer('vocab', 'building the vocabulary... '):
        vocab = Vocab(hps.vocab_file, hps.vocab_size)
    print('└─ vocab size: %s' % vocab.size)

    with timer('loader', 'building the dataloader... '):
        loader = build_loader(vocab, hps)

    with timer('model', 'building the model... '):
        if 'hypo' in hps.mode:
            rnn = HypoModel(vocab, hps, rnn_type=hps.rnn_type)
        else:
            rnn = SNLIRNN(vocab, hps, rnn_type=hps.rnn_type)

    if hps.use_cuda:
        with timer('cuda', '└─ copying the model to gpu... '):
            rnn = rnn.cuda()

    model = Model(rnn, hps.model_name, hps.log_dir, hps=hps)

    if hps.print_model:
        print(rnn)

    mode = hps.mode.replace('hypo', '')
    if mode == 'train':
        train(model, vocab, loader['train'], loader['val'], hps)
    elif mode == 'val':
        test(model, vocab, loader['val'], hps)
    elif mode == 'test':
        test(model, vocab, loader['test'], hps)
    else:
        raise ValueError('Unknown mode: %s', hps.mode)


if __name__ == '__main__':
    main(hps)
