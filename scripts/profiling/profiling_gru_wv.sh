#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 100-100 WV-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 100 --pretrained --batch-size 64 --verbose 4