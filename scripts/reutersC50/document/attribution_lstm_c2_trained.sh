#!/usr/bin/env bash

python3 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 20 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C2-20 keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C2)" --rnn-type LSTM --epoch 150 --batch-size 128 --verbose 4 --cuda

