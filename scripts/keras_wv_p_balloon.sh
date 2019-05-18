#!/bin/bash

python2.7 model_balloon_keras.py --hidden-size 100 --cell-size 100 --embedding-size 400 --feature wv --precomputed-features pca100 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 100-100 WV-P ESN-100 keras" --description "Authorship Attribution 15 authors Balloon 100 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 100" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 100 --cell-size 100 --embedding-size 550 --feature wv --precomputed-features pca250 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 100-100 WV-P ESN-250 keras" --description "Authorship Attribution 15 authors Balloon 100 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 250" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 100 --cell-size 100 --embedding-size 800 --feature wv --precomputed-features pca500 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 100-100 WV-P ESN-500 keras" --description "Authorship Attribution 15 authors Balloon 100 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 500" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda



python2.7 model_balloon_keras.py --hidden-size 200 --cell-size 200 --embedding-size 400 --feature wv --precomputed-features pca100 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 200-200 WV-P ESN-100 keras" --description "Authorship Attribution 15 authors Balloon 200 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 100" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 200 --cell-size 200 --embedding-size 550 --feature wv --precomputed-features pca250 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 200-200 WV-P ESN-250 keras" --description "Authorship Attribution 15 authors Balloon 200 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 250" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 200 --cell-size 200 --embedding-size 800 --feature wv --precomputed-features pca500 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 200-200 WV-P ESN-500 keras" --description "Authorship Attribution 15 authors Balloon 200 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 500" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda



python2.7 model_balloon_keras.py --hidden-size 300 --cell-size 300 --embedding-size 400 --feature wv --precomputed-features pca100 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 300-300 WV-P ESN-100 keras" --description "Authorship Attribution 15 authors Balloon 300 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 100" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 300 --cell-size 300 --embedding-size 550 --feature wv --precomputed-features pca250 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 300-300 WV-P ESN-250 keras" --description "Authorship Attribution 15 authors Balloon 300 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 250" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda

python2.7 model_balloon_keras.py --hidden-size 300 --cell-size 300 --embedding-size 800 --feature wv --precomputed-features pca500 --learning-window 40 --max-length 40 --k 10 --output outputs/ --name "AA Na15 Balloon 300-300 WV-P ESN-500 keras" --description "Authorship Attribution 15 authors Balloon 300 hidden neurons, Word Vectors (WV) pretrained with Glove, ESN reduced to 500" --rnn-type LSTM --epoch 250 --batch-size 64 --pretrained --verbose 4 --cuda
