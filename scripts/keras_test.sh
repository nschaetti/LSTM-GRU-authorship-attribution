#!/bin/bash

python2.7 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --n-authors 50 --k 10 --output outputs/ --name "AA Na50 LSTM 100-100 WV-P keras" --description "Authorship Attribution 50 authors LSTM 100 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 5 --pretrained --verbose 4 --cuda

# python2.7 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --n-authors 50 --k 10 --output outputs/ --name "AA Na50 LSTM 200-200 WV-P keras" --description "Authorship Attribution 50 authors LSTM 200 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 5 --pretrained --verbose 4 --cuda

# python2.7 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --n-authors 50 --k 10 --inverse-dev-test --output outputs/ --name "AA Na50 LSTM 300-300 WV-P keras" --description "Authorship Attribution 50 authors LSTM 300 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 5 --pretrained --verbose 4 --cuda
