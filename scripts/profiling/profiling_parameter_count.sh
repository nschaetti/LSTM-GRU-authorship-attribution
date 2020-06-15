#!/usr/bin/env bash

# GRU WV-P 100 units
python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --xpname "AP GRU 100-100 WV-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 100 --pretrained --batch-size 64 --verbose 4

# GRU WV-T 25 units
python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --learning-rate 0.0001 --output-dropout 0.5 --k 10 --output outputs/profiling/ --xpname "AP GRU 25-25 WV-T" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Word Vectors (WV) trained" --rnn-type GRU --epoch 200 --batch-size 128 --verbose 4

# GRU C3-T 25 units
python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 25-25 C3-T" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Character trigram (C3) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4

# GRU C2-T 25 units
python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 25-25 C2-T" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Character bigram (C2) trained" --rnn-type GRU --epoch 100 --batch-size 64 --verbose 4

# GRU C1-T 50 units
python3 model_rnn_profiling.py --hidden-size 50 --cell-size 50 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 50-50 C1-T" --description "Authorship Profiling PAN17 GRU 50 hidden neurons, Character unigram (C1) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --epoch 300 --batch-size 64 --verbose 4

# GRU C3-P 25 units
python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 25-25 C3-P" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Character trigram (C3) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c3_cx2_d60.p --epoch 100 --pretrained --batch-size 64 --verbose 4

# GRU C2-P 25 units
python3 model_rnn_profiling.py --hidden-size 25 --cell-size 25 --embedding-size 10 --feature c2 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 25-25 C2-P" --description "Authorship Profiling PAN17 GRU 25 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --epoch 100 --pretrained --batch-size 64 --verbose 4

# GRU C1-P 100 units
python3 model_rnn_profiling.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --output-dropout 0.5 --learning-rate 0.0001 --k 10 --output outputs/profiling/ --xpname "AP GRU 100-100 C1-P" --description "Authorship Profiling PAN17 GRU 100 hidden neurons, Character unigram (C1) pre-trained" --rnn-type GRU --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --epoch 100 --pretrained --batch-size 64 --verbose 4
