#!/bin/bash

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 25 --cell-size 25 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.25 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC ASIMOV GRU 25-25 WV-T DOO 0.5" --description "Authorship Verification SFGram Document GRU 25 hidden neurons, Word Vectors (WV) trained, Dropout-output 0.5, Zero weight 1" --rnn-type GRU --epoch 60 --batch-size 512 --verbose 4 --cuda
python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.25 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC ASIMOV GRU 50-50 WV-T DOO 0.5" --description "Authorship Verification SFGram Document GRU 50 hidden neurons, Word Vectors (WV) trained, Dropout-output 0.5, Zero weight 1" --rnn-type GRU --epoch 60 --batch-size 512 --verbose 4 --cuda
python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 100 --cell-size 100 --embedding-size 100 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.25 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC ASIMOV GRU 100-100 WV-T DOO 0.5" --description "Authorship Verification SFGram Document GRU 100 hidden neurons, Word Vectors (WV) trained, Dropout-output 0.5, Zero weight 1" --rnn-type GRU --epoch 60 --batch-size 512 --verbose 4 --cuda
