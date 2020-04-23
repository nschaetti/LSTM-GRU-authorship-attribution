#!/usr/bin/env bash

python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.0001 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.0001" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.0001, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.001 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.001" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.001, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.01 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.01" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.01, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.1 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.1" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.1, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.2 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.2" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.2, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.3 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.3" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.3, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.4 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.4" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.4, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.5 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.5" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.5, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.6 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.6" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.6, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.7 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.7" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.7, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.8 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.8" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.8, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 0.9 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 0.9" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 0.9, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4
python3 model_esn_profiling.py --hidden-size 1000 --feature c1 --leaky-rate 1.0 --k 10 --output outputs/profiling/ --name "AP ESN 1000 C1 LK 1.0" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons leak rate 1.0, Character unigram (C1)" --embedding-path ~/Projets/TURING/Recherches/PhD/LSTM-GRU-authorship-attribution/embeddings/char_embedding/c1_cx2_d10.p --pretrained --verbose 4

