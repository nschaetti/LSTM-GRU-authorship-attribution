#!/usr/bin/env bash

python3 model_rnn_keras.py --fold 8 --hidden-size 300 --cell-size 300 --embedding-size 30 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C3-30 keras Fold 8" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C3)" --rnn-type LSTM --epoch 140 --batch-size 128 --verbose 4 --cuda