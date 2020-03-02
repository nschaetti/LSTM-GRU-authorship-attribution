#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 20 --cell-size 20 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 20-20 C2-T" --description "Authorship Profiling PAN17 GRU 20 hidden neurons, Character bigram (C2) trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --epoch 5 --batch-size 32 --verbose 4 --cuda
