#!/usr/bin/env bash

# LSTM-C3 trained
python3 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 30 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C3-30 keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C3)" --rnn-type LSTM --epoch 140 --batch-size 64 --verbose 4 --cuda

# LSTM-WV pretrained
python3 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 WV-P keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda
