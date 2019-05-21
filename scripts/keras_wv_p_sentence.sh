#!/bin/bash

python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 100-100 WV-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 100 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4

# python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 300-300 WV-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 300 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4

python2.7 model_rnn_keras_sentence.py --hidden-size 500 --cell-size 500 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 500-500 WV-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 500 hidden neurons, Word Vectors (WV) pretrained with Glove" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4
