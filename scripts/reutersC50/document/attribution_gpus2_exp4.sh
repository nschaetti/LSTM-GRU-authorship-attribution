#!/usr/bin/env bash

# LSTM C3 train 300
python3 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 30 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --xpname "AA Na15 LSTM 300-300 c3-30 keras" --description "Authorship Attribution 15 authors LSTM 300 hidden neurons, Character (C3)" --rnn-type LSTM --epoch 140 --batch-size 64 --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p
