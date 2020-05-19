#!/usr/bin/env bash

# GRU-WV pretrained 200
python3 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output output/ --xpname "AA Na15 GRU 200-200 WV-P keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type GRU --epoch 150 --batch-size 64 --pretrained --verbose 4

# GRU-WV trained 50
python3 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output output/ --xpname "AA Na15 GRU 100-100 WV-50 keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Word Vectors (WV) trained size 50" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4 --cuda

