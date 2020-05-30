#!/usr/bin/env bash

python3 model_rnn_keras.py --fold 9 --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 WV-P keras Fold 9" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 128 --pretrained --verbose 4

