#!/usr/bin/env bash

python3 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 C3-60 keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Character (C3)" --rnn-type LSTM --epoch 140 --batch-size 64 --verbose 4 --cuda

