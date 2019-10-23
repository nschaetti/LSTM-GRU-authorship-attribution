#!/bin/bash

python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 100-100 C3-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 100 hidden neurons, Character (C3) pretrained" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c3_cx2_d60.p

python2.7 model_rnn_keras_sentence.py --hidden-size 200 --cell-size 200 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 200-200 C3-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 200 hidden neurons, Character (C3) pretrained" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c3_cx2_d60.p

python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 LSTM 300-300 C3-P keras" --description "Authorship Attribution Sentence 15 authors LSTM 300 hidden neurons, Character (C3) pretrained" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c3_cx2_d60.p