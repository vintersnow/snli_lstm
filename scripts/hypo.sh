#!/bin/bash

python src/main.py \
  --use_cuda \
  --mode hypotrain \
  --print_model \
  --num_iters 100000 \
  --vocab_size 0 \
  --max_steps 50 \
  --check_steps 500 \
  --ckpt_steps 2000 \
  --summary_steps 100 \
  --batch_size 64 \
  --num_layers 1 \
  --dropout 0.0 \
  --opt adam \
  --init_lr 0.001 \
  --embd_size 300 \
  --hidden_size 256 \
  --model_name snli_hypo_adam001_ly1e300h256b64d0 \
  --pre_embd vector.npy \
  $*
