#!/bin/bash

# Word pretrained ASIMOV
python3 model_rnn_sfgram_document.py --author0 ASIMOV --hidden-size 50  --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output outputs/sfgram/ --name "SFG DOC ASIMOV GRU 50-50 WV-P" --description "Authorship Verification Document ASIMOV SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained" --rnn-type GRU --epoch 10 --pretrained --batch-size 256 --verbose 4 --cuda
