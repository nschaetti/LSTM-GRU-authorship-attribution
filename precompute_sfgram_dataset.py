#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : model_rnn_profiling.py
# Description : Profiling on PAN17 with LSTM and GRU.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import torch.utils.data
from tools import argument_parsing, dataset, functions, features, settings
from tools import rnn as rnn_func
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

#region INIT

# Seed
torch.manual_seed(1)

# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, \
    num_layers, dropout, output_dropout = functions.get_params(space)

    # Load SFGram dataset
    sfgram_dataset, sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test = dataset.load_sfgram_dataset(
        block_length=40,
        batch_size=args.batch_size,
        author='SILVERBERG',
        load_type=feature + ("" if args.pretrained else "T")
    )

    # Print dataset information
    xp.write("Dataset length : {}".format(len(sfgram_dataset)), log_level=0)
    xp.write("Number of texts : {}".format(len(sfgram_dataset.texts)), log_level=0)

    # Choose the right transformer
    sfgram_dataset.transform = features.create_transformer(
        feature,
        args.pretrained,
        args.embedding_path,
        lang
    )

    # Precompute the dataset
    sfgram_dataset.precompute_documents("./sfgram_blocks")
# end for
