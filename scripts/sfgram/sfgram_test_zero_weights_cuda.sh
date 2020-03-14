#!/bin/bash

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 1" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 1" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.25 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 1.25" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 1.25" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.5 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 1.5" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 1.5" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 2 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 2" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 2" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 3 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 3" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 3" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 4 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 4" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 4" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 5 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 5" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 5" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 6 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 6" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 6" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 7 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 7" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 7" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 8 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 8" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 8" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda

python3 model_rnn_sfgram.py --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 9 --output outputs/sfgram/ --name "SFG GRU 50-50 WV-P ZW 9" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Zero weight 9" --rnn-type GRU --epoch 300 --pretrained --batch-size 1024 --verbose 4 --cuda
