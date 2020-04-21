#!/bin/bash

python3 model_esn_profiling.py --hidden-size 1000 --feature wv --leaky-rate 1.0 --k 10 --output outputs/profiling/ --name "AP ESN 1000 WV" --description "Authorship Profiling PAN17 ESN 1000 hidden neurons, Word vectors (WV)" --pretrained --verbose 4
