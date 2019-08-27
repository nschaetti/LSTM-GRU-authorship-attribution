#!/bin/bash

python model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 POS-10 keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 POS-10 keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 POS-10 keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 100-100 POS-10 keras" --description "Authorship Attribution Sentence 15 authors GRU 100 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras_sentence.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 200-200 POS-10 keras" --description "Authorship Attribution Sentence 15 authors GRU 200 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

python model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 300-300 POS-10 keras" --description "Authorship Attribution Sentence 15 authors GRU 300 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type GRU --epoch 150 --batch-size 64 --verbose 4 --cuda

