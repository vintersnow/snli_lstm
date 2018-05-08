#!/bin/bash

python src/main.py \
  --use_cuda \
  --mode train \
  --print_model \
  --num_iters 100000 \
  --vocab_size 0 \
  --max_steps 50 \
  --check_steps 300 \
  --ckpt_steps 300 \
  --summary_steps 100 \
  --batch_size 64 \
  --num_layers 1 \
  --dropout 0.0 \
  --opt adam \
  --init_lr 0.001 \
  --embd_size 300 \
  --hidden_size 256 \
  --pre_embd vector.npy \
  --model_name snli_adam001_ly1e300h512b64d0_embd \
  $*
