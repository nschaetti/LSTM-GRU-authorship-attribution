#!/bin/bash

# Word pretrained ASIMOV
# python3 model_rnn_sfgram.py --author0 ASIMOV --hidden-size 25  --cell-size 25 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG ASIMOV GRU 25-25 C2-P" --description "Authorship Verification ASIMOV SFGram GRU 25 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 60 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
# python3 model_rnn_sfgram.py --author0 ASIMOV --hidden-size 50  --cell-size 50 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG ASIMOV GRU 50-50 C2-P" --description "Authorship Verification ASIMOV SFGram GRU 50 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 65 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
# python3 model_rnn_sfgram.py --author0 ASIMOV --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG ASIMOV GRU 100-100 C2-P" --description "Authorship Verification ASIMOV SFGram GRU 100 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 70 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda

# Word pretrained DICK
# python3 model_rnn_sfgram.py --author0 DICK --hidden-size 25  --cell-size 25 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 25-25 C2-P" --description "Authorship Verification DICK SFGram GRU 25 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 60 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
python3 model_rnn_sfgram.py --author0 DICK --hidden-size 50  --cell-size 50 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 50-50 C2-P" --description "Authorship Verification DICK SFGram GRU 50 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 65 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
python3 model_rnn_sfgram.py --author0 DICK --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DICK GRU 100-100 C2-P" --description "Authorship Verification DICK SFGram GRU 100 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 70 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda

# Word pretrained SILVERBERG
# python3 model_rnn_sfgram.py --author0 SILVERBERG --hidden-size 25  --cell-size 25 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 25-25 C2-P" --description "Authorship Verification SILVERBERG SFGram GRU 25 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 60 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
# python3 model_rnn_sfgram.py --author0 SILVERBERG --hidden-size 50  --cell-size 50 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 50-50 C2-P" --description "Authorship Verification SILVERBERG SFGram GRU 50 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 65 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda
# python3 model_rnn_sfgram.py --author0 SILVERBERG --hidden-size 100 --cell-size 100 --embedding-size 30 --feature c2 --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG SILVERBERG GRU 100-100 C2-P" --description "Authorship Verification SILVERBERG SFGram GRU 100 hidden neurons, Character bigram (C2) pre-trained" --rnn-type GRU --epoch 70 --pretrained --batch-size 256 --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c2_cx2_d30.p --verbose 4 --cuda

