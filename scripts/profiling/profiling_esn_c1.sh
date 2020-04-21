#!/usr/bin/env bash

python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 1.0 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4

