#!/bin/bash

python2.7 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 FW-P keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras.py --hidden-size 20 --cell-size 200 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 FW-P keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 FW-P keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 FW-20 keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda

python2.7 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 FW-20 keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda

python2.7 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 FW-20 keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda


python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 100-100 FW-P keras" --description "Authorship Attribution Sentence 15 authors GRU 100 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras_sentence.py --hidden-size 20 --cell-size 200 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 200-200 FW-P keras" --description "Authorship Attribution Sentence 15 authors GRU 200 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 300 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 300-300 FW-P keras" --description "Authorship Attribution Sentence 15 authors GRU 300 hidden neurons, Function Words (FW) pretrained with Glove" --rnn-type GRU --epoch 200 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 100-100 FW-20 keras" --description "Authorship Attribution Sentence 15 authors GRU 100 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda

python2.7 model_rnn_keras_sentence.py --hidden-size 200 --cell-size 200 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 200-200 FW-20 keras" --description "Authorship Attribution Sentence 15 authors GRU 200 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda

python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 20 --feature fw --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 300-300 FW-20 keras" --description "Authorship Attribution Sentence 15 authors GRU 300 hidden neurons, Function Words (FW)" --rnn-type GRU --epoch 200 --batch-size 64 --verbose 4 --cuda
