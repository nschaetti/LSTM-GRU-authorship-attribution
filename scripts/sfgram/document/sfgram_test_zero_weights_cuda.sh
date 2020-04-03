#!/bin/bash

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 1" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 1" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.25 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 1.25" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 1.25" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 1.5 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 1.5" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 1.5" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 2 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 2" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 2" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 3 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 3" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 3" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 4 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 4" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 4" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 5 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 5" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 5" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 6 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 6" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 6" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 7 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 7" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 7" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 8 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 8" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 8" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda

python3 model_rnn_sfgram_document.py --author0 ASIMOV  --hidden-size 50 --cell-size 50 --embedding-size 300 --feature wv --learning-window 40 --max-length 40 --k 5 --zero-weight 9 --output-dropout 0.5 --output outputs/sfgram/ --name "SFG DOC GRU 50-50 WV-P DOO 0.5 ZW 9" --description "Authorship Verification SFGram GRU 50 hidden neurons, Word Vectors (WV) pre-trained, Dropout-output 0.5, Zero weight 9" --rnn-type GRU --epoch 50 --pretrained --batch-size 512 --verbose 4 --cuda