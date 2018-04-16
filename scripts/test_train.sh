
#!/bin/bash

python src/main.py \
  --mode train \
  --num_iters 1 \
  --print_model \
  --vocab_size 0 \
  --test_data small_test.json\
  --train_data small_train.json\
  --val_data small_train.json\
  --max_steps 100 \
  --batch_size 11 \
  --num_layers 1 \
  --summary_steps 1 \
  --embd_size 300 \
  --check_steps 1 \
  --opt adam \
  --init_lr 0.001 \
  --val_num 10 \
  --pre_embd vector.npy \
  $*
