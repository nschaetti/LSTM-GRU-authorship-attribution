#!/bin/bash

python3 model_rnn_profiling_single.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 50-50 Single WV-P" --description "Authorship Profiling PAN17 GRU 50 hidden neurons, Single Tweet, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 50 --pretrained --batch-size 128 --verbose 4 --cuda
