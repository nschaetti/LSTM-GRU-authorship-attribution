#!/usr/bin/env bash

# C3 pretrained DICK
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 25  --cell-size 25 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 25-25 C3-P INVERSE" --description "Authorship Verification DICK SFGram GRU 25 hidden neurons, Character trigram (C3) pre-trained, inverse" --rnn-type GRU --epoch 60 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c3_cx2_d60.p --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 50  --cell-size 50 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 50-50 C3-P INVERSE" --description "Authorship Verification DICK SFGram GRU 50 hidden neurons, Character trigram (C3) pre-trained, inverse" --rnn-type GRU --epoch 65 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c3_cx2_d60.p --verbose 4 --cuda
python3 model_rnn_sfgram.py --inverse-dev-test --author0 DICK --hidden-size 100 --cell-size 100 --embedding-size 60 --feature c3 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 100-100 C3-P INVERSE" --description "Authorship Verification DICK SFGram GRU 100 hidden neurons, Character trigram (C3) pre-trained, inverse" --rnn-type GRU --epoch 70 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c3_cx2_d60.p --verbose 4 --cuda

