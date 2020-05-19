#!/usr/bin/env bash

# C1 trained SILVERBERG
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 25  --cell-size 25 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 25-25 C1-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 25 hidden neurons, Character unigram (C1) trained, inverse" --rnn-type GRU --epoch 50 --batch-size 256  --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 50  --cell-size 50 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 50-50 C1-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 50 hidden neurons, Character unigram (C1) trained, inverse" --rnn-type GRU --epoch 55 --batch-size 256 --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 100 --cell-size 100 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 100-100 C1-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 100 hidden neurons, Character unigram (C1) trained, inverse" --rnn-type GRU --epoch 60 --batch-size 256 --verbose 4 --cuda
