#!/usr/bin/env bash

# GRU-WV pretrained 210
python3 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --xpname "AA Na15 GRU 100-100 WV-P keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type GRU --epoch 150 --batch-size 64 --pretrained --verbose 4

