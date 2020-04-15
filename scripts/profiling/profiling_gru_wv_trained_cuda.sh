#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 50 --cell-size 50 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 50-50 WV-T" --description "Authorship Profiling PAN17 GRU 50 hidden neurons, Word Vectors (WV) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4 --cuda
# python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 100-100 WV-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 50 --pretrained --batch-size 32 --verbose 4 --cuda
