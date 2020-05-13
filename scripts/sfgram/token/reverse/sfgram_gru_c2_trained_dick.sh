#!/usr/bin/env bash

# C2 trained DICK
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 25  --cell-size 25 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 25-25 C2-T INVERSE" --description "Authorship Verification DICK SFGram GRU 25 hidden neurons, Character bigram (C2) trained, inverse" --rnn-type GRU --epoch 50 --batch-size 256 --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 50  --cell-size 50 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 50-50 C2-T INVERSE" --description "Authorship Verification DICK SFGram GRU 50 hidden neurons, Character bigram (C2) trained, inverse" --rnn-type GRU --epoch 55 --batch-size 256 --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 100-100 C2-T INVERSE" --description "Authorship Verification DICK SFGram GRU 100 hidden neurons, Character bigram (C2) trained, inverse" --rnn-type GRU --epoch 60 --batch-size 256 --verbose 4 --cuda

