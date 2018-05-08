#!/bin/bash

python src/main.py \
  --use_cuda \
  --mode hypoval \
  --print_model \
  --vocab_size 0 \
  --val_num 0 \
  --max_steps 50 \
  --batch_size 64 \
  --num_layers 1 \
  --dropout 0.0 \
  --embd_size 300 \
  --hidden_size 256 \
  --model_name snli_hypo_adam001_ly1e300h256b64d0 \
  --save_pred \
  $*
