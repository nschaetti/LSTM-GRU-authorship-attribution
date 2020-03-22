#!/bin/bash

python3 model_rnn_profiling_single.py --hidden-size 20 --cell-size 20 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP GRU 20-20 Single C1-P" --description "Authorship Profiling PAN17 GRU 20 hidden neurons, Single Tweet, Character unigram (C1) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --epoch 150 --batch-size 512 --verbose 4 --cuda
