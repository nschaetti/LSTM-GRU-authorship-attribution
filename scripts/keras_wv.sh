#!/bin/bash

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 WV-50 keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Word Vectors (WV) trained size 50" --rnn-type LSTM --epoch 100 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 WV-50 keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Word Vectors (WV) trained size 50" --rnn-type LSTM --epoch 100 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 50 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 WV-50 keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Word Vectors (WV) trained size 50" --rnn-type LSTM --epoch 100 --batch-size 32 --verbose 4 --cuda
