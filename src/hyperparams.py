from argparse import ArgumentParser
import torch

parser = ArgumentParser('python main.py')

# Data
parser.add_argument('--vocab_size', type=int, default=50000,
                    help='max number of vocabulary')
parser.add_argument('--vocab_file', type=str, default='data/vocab',
                    help='path to vocabulary file')
parser.add_argument('--data_path', type=str, default='data/convert',
                    help='path to data file or directory')
parser.add_argument('--train_data', type=str, default='conv2_train.json',
                    help='')
parser.add_argument('--val_data', type=str, default='conv2_dev.json',
                    help='')
parser.add_argument('--test_data', type=str, default='conv2_test.json',
                    help='')
parser.add_argument('--single_pass', action='store_true',
                    help='If the flag is setted, applay example only once')

# Model
parser.add_argument('--max_steps', type=int, default=50,
                    help='max length for encoder')
parser.add_argument('--embd_size', type=int, default=256,
                    help='size of word embedding')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='size of hidden units')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of leyers in the encoder')
parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='RNN function. GRU or RNN or LSTM')
parser.add_argument('--word_freq', type=int, default=0,
                    help='minmum word frequency')
parser.add_argument('--bidirectional', action='store_true', help='')
parser.add_argument('--pre_embd', type=str, default='', help='')
parser.add_argument('--no_train_embd', action='store_true', help='')

# Training
parser.add_argument('--num_iters', type=int, default=100,
                    help='number of training iterations')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--init_lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=2.0,
                    help='gradient clipping norm size')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout ratio for encoder and decoder')
parser.add_argument('--opt', type=str, default='adam',
                    help='optimizer method')
parser.add_argument('--start_step', type=int, default=1,
                    help='initial step number')

# Summary and checkpoint
parser.add_argument('--summary_steps', type=int, default=10,
                    help='interval for reporting summary')
parser.add_argument('--store_summary', action='store_true',
                    help='store summary')
parser.add_argument('--check_steps', type=int, default=1000,
                    help='interval for reporting score')
parser.add_argument('--ckpt_steps', type=int, default=5000,
                    help='interval for reporting summary')
parser.add_argument('--val_num', type=int, default=10000,
                    help='validation samples')
parser.add_argument('--model_name', type=str, default='default',
                    help='name of the model ')
parser.add_argument('--restore', type=str, default=None,
                    help='best or latest')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory to save ckpt, prediction, summary')


# Other
parser.add_argument('--use_cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--print_model', action='store_true',
                    help='print model before run')
parser.add_argument('--mode', type=str, default='train',
                    help='train or decode')
# parser.add_argument('--setting', type=str, default=None,
#                     help='setting file (json)')
parser.add_argument('--save_pred', action='store_true',
                    help='save model prediction (only use in val and test)')


hps = parser.parse_args()

hps.use_cuda = hps.use_cuda and torch.cuda.is_available()
hps.torch = True
