#!/bin/bash

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 100-100 WV-P keras" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4
python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --inverse-dev-test --output outputs/ --name "AA Na15 LSTM 100-100 WV-P keras Inverse" --description "Authorship Attribution 15 authors LSTM 100 hidden neurons, Word Vectors (WV) pretrained with Glove Inverse Dev Test" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 WV-P keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4
python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --inverse-dev-test --output outputs/ --name "AA Na15 LSTM 300-300 WV-P keras Inverse" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Word Vectors (WV) pretrained with Glove Inverse Dev Test" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4

python model_rnn_keras.py --hidden-size 1000 --cell-size 1000 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 1000-1000 WV-P keras" --description "Authorship Attribution 15 authors LSTM 1000 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4
python model_rnn_keras.py --hidden-size 1000 --cell-size 1000 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --inverse-dev-test --output outputs/ --name "AA Na15 LSTM 1000-1000 WV-P keras Inverse" --description "Authorship Attribution 15 authors LSTM 1000 hidden neurons, Word Vectors (WV) pretrained with Glove Inverse Dev Test" --rnn-type LSTM --epoch 250 --batch-size 32 --pretrained --verbose 4
