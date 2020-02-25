#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 100-100 C1-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Character unigram (C1) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --epoch 1 --pretrained --batch-size 48 --verbose 4 --cuda
