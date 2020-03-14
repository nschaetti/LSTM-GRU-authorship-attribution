#!/bin/bash

# Word pretrained
python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.5 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 5 --pretrained --batch-size 1024 --verbose 4 --cuda
