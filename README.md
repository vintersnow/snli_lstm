# SNLI prediction with series connected LSTM

# Requirements

- torch(0.4~)
- tensorboardX
- (optional) spacy

Run below command before training

```
$ git submodule update --init --recursive
```

## How to run

### Download the data and modified

Download SNLI dataset from [the official site](https://nlp.stanford.edu/projects/snli/) and unzip the file at the root directory of this repository.
run `convert_data.py` to convert the data.

```
$ python convert_data.py --data_path {unzip directory name: default is `data`}
```

### train
Run sample script.

```
$ ./scripts/train.sh --store_summary
```

### validation

```
$ ./scripts/val.sh
```

## tensorboard

```
tensorboard --logdir logs
```

# Use pretrained embedding

Using spacy for pretrained embedding

```
$ python make_embd.py
$ ./scripts/train.sh --store_summary --pre_embd vector.npy
```
