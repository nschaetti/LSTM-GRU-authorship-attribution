#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 50 --cell-size 50 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --output-dropout 0.25 --k 10 --output outputs/profiling/ --name "AP GRU 50-50 C2-T" --description "Authorship Profiling PAN17 GRU 50 hidden neurons, Character bigram (C2) trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --epoch 100 --batch-size 64 --verbose 4 --cuda
