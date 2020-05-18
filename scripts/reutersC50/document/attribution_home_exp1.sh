#!/usr/bin/env bash

# GRU-WV trained 50
python3 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 WV-50 keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Word Vectors (WV) trained size 50" --rnn-type GRU --epoch 100 --batch-size 128 --verbose 4 --cuda

