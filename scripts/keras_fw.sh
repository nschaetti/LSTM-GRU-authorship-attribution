#!/bin/bash

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 FW-P keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 20 --cell-size 200 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 FW-P keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 FW-P keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 FW-P keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 20 --cell-size 200 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 FW-P keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 FW-P keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type LSTM --epoch 200 --batch-size 64 --verbose 4 --cuda
