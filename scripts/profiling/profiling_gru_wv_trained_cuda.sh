#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 20 --cell-size 20 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 20-20 WV-T" --description "Authorship Profiling PAN17 GRU 20 hidden neurons, Word Vectors (WV) trained" --rnn-type GRU --epoch 5 --batch-size 32 --verbose 4 --cuda

# python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 100-100 WV-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 50 --pretrained --batch-size 32 --verbose 4 --cuda
