#!/bin/bash

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 C1-P keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Character (C1) pretrained" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 C1-P keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Character (C1) pretrained" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C1-P keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C1) pretrained" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 C1-10 keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Character (C1)" --rnn-type LSTM --epoch 250 --batch-size 23 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 C1-10 keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Character (C1)" --rnn-type LSTM --epoch 250 --batch-size 32 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature C1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C1-10 keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C1)" --rnn-type LSTM --epoch 250 --batch-size 32 --verbose 4 --cuda
