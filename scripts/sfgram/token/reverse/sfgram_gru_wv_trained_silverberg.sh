#!/usr/bin/env bash

# Word pretrained SILVERBERG
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 25  --cell-size 25 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 25-25 WV-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 25 hidden neurons, Word Vectors (WV) trained, inverse" --rnn-type GRU --epoch 100 --batch-size 256 --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 50  --cell-size 50 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 50-50 WV-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 50 hidden neurons, Word Vectors (WV) trained, inverse" --rnn-type GRU --epoch 100 --batch-size 256 --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 SILVERBERG --hidden-size 100 --cell-size 100 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 100-100 WV-T INVERSE" --description "Authorship Verification SILVERBERG SFGram GRU 100 hidden neurons, Word Vectors (WV) trained, inverse" --rnn-type GRU --epoch 100 --batch-size 256 --verbose 4 --cuda


