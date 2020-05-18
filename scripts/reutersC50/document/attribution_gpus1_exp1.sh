#!/usr/bin/env bash

# LSTM-POS trained
python3 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 50 --feature pos --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 200-200 POS-10 keras" --description "Authorship Attribution 15 authors LSTM 200 hidden neurons, Part-of-Speech (POS) trained size 10" --rnn-type LSTM --epoch 150 --batch-size 128 --verbose 4 --cuda

# LSTM-C1 pre-trained
python3 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 LSTM 300-300 C1-P keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C1) pretrained" --rnn-type LSTM --epoch 150 --batch-size 128 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p
