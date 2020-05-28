#!/usr/bin/env bash

python3 model_rnn_keras.py --fold 8 --hidden-size 200 --cell-size 200 --embedding-size 30 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 C3-30 keras Fold 8" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Character (C3)" --rnn-type LSTM --epoch 140 --batch-size 128 --verbose 4 --cuda
