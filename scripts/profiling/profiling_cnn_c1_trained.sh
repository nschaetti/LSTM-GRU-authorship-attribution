#!/bin/bash

python3 model_cnn_c1_trained.py --embedding-size 10 --feature c1 --learning-window 40 --max-length 40 --k 10 --output outputs/profiling/ --name "AP CNN 50C Single C1-P" --description "Authorship Profiling PAN17 CNN 50 channels, Single Tweet, Character unigram (C1) pre-trained" --epoch 150 --batch-size 128 --verbose 4 --cuda
