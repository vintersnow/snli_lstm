#!/bin/sh

rsync -a --progress \
  --exclude __pycache__\
  --exclude .git \
  --exclude data \
  --exclude *.log \
  --exclude *.csv \
  --exclude logs \
  -e ssh ~/Projects/ut/nlp/snli_lstm kobe:~/projects/
