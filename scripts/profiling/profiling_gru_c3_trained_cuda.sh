#!/bin/bash

python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --name "AP GRU 25-25 C3-T" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Character trigram (C3) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4 --cuda
python3 model_rnn_profiling.py --hidden-size 50 --cell-size 50 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --name "AP GRU 50-50 C3-T" --description "Authorship Profiling PAN17 GRU 50 hidden neurons, Character trigram (C3) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4 --cuda
python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --name "AP GRU 100-100 C3-T" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Character trigram (C3) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4 --cuda
