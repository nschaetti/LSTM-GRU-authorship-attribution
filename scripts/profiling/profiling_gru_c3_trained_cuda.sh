#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 20 --cell-size 20 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 20-20 C3-T" --description "Authorship Profiling PAN17 GRU 20 hidden neurons, Character trigram (C3) trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c3_cx2_d60.p --epoch 5 --batch-size 32 --verbose 4 --cuda
