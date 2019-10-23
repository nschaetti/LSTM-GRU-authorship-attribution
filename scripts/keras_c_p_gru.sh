#!/bin/bash

#### C1

# Document

# python2.7 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 C1-P keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p

# python2.7 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 C1-P keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 16 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p

# python2.7 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 C1-P keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 16 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p

# Sentence

# python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 100-100 C1-P keras" --description "Authorship Attribution Sentence 15 authors GRU 100 hidden neurons, Character (C1) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p

python2.7 model_rnn_keras_sentence.py --hidden-size 200 --cell-size 200 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 200-200 C1-P keras" --description "Authorship Attribution Sentence 15 authors GRU 200 hidden neurons, Character (C1) pretrained" --rnn-type GRU --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p

python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 300-300 C1-P keras" --description "Authorship Attribution Sentence 15 authors GRU 300 hidden neurons, Character (C1) pretrained" --rnn-type GRU --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c1_cx2_d10.p


#### C2

# Document

# python2.7 model_rnn_keras.py --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 100-100 C2-P keras" --description "Authorship Attribution 15 authors GRU 100 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

# python2.7 model_rnn_keras.py --hidden-size 200 --cell-size 200 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 200-200 C2-P keras" --description "Authorship Attribution 15 authors GRU 200 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

python2.7 model_rnn_keras.py --hidden-size 300 --cell-size 300 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 GRU 300-300 C2-P keras" --description "Authorship Attribution 15 authors GRU 300 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 32 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

# Sentence

# python2.7 model_rnn_keras_sentence.py --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 100-100 C2-P keras" --description "Authorship Attribution Sentence 15 authors GRU 100 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

# python2.7 model_rnn_keras_sentence.py --hidden-size 200 --cell-size 200 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 200-200 C2-P keras" --description "Authorship Attribution Sentence 15 authors GRU 200 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

# python2.7 model_rnn_keras_sentence.py --hidden-size 300 --cell-size 300 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AAS Na15 GRU 300-300 C2-P keras" --description "Authorship Attribution Sentence 15 authors GRU 300 hidden neurons, Character (C2) pretrained" --rnn-type GRU --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda --embedding-path ~/Projets/TURING/Recherches/MLE/PhD/Reuters-C50-Authorship-Attribution/embeddings/char_embedding/c2_cx2_d30.p

