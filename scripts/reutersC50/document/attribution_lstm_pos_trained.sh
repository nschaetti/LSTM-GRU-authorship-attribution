#!/usr/bin/env bash

# LSTM-POS trained
python3 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 50 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 POS-10 keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type LSTM --epoch 150 --batch-size 128 --verbose 4 --cuda

