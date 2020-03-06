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
    sfgram_dataset, sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test = dataset.load_sfgram_precomputed_dataset(
        block_length=40,
        batch_size=args.batch_size,
        author='ASIMOV',
        pretrained=args.pretrained,
        feature=feature
    )

    # Print dataset information
    xp.write("Dataset length : {}".format(len(sfgram_dataset.data_length)), log_level=0)
    xp.write("Number of texts : {}".format(len(sfgram_dataset.dataset_size)), log_level=0)
    xp.write("Number of trues : {}".format(len(sfgram_dataset.trues_count)), log_level=0)

    # Set experience state
    xp.set_state(space)

    # For each sample
    for n in range(args.n_samples):
        # Set sample state
        xp.set_sample_state(n)

        # Average
        average_k_fold = np.array([])

        # For each fold
        for k in range(5):
            # Change fold state
            xp.set_fold_state(k)
            sfgram_loader_train.dataset.set_fold(k)
            sfgram_loader_test.dataset.set_fold(k)

            # Model
            rnn = rnn_func.create_verification_model(
                feature=feature,
                pretrained=args.pretrained,
                cuda=args.cuda,
                embedding_dim=embedding_size,
                hidden_dim=hidden_size,
                vocab_size=settings.verification_voc_size[feature],
                rnn_type=rnn_type,
                num_layers=num_layers,
                dropout=dropout,
                output_dropout=output_dropout,
                batch_size=args.batch_size
            )

            # Optimizer
            # optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
            optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

            # Best model
            best_model = None

            # Loss function
            loss_function = nn.MSELoss()

            # For each epoch
            for epoch in range(args.epoch):
                # Training, validation losses
                training_loss = 0.0
                training_total = 0.0
                validation_loss = 0.0
                validation_total = 0.0

                # Number of samples
                n_samples = len(sfgram_loader_train)

                # RNN in training mode
                rnn.train()

                # Go through the training set
                for i, data in enumerate(sfgram_loader_train):
                    # Data
                    inputs, outputs = data
                    print(inputs.size())
                    print(outputs.size())
                    exit()
                # end for

                # Evaluation mode
                rnn.eval()

                # Go through the validation set
                for i, data in enumerate(sfgram_loader_dev):
                    # Data
                    inputs, outputs = data
                # end for

                # Show loss
                print("epoch {}, training loss {} ({}), validation loss {} ({})".format(
                    epoch,
                    training_loss / training_total,
                    training_total,
                    validation_loss / validation_total,
                    validation_total
                ))
            # end for

            # Test loss
            test_loss = 0.0
            test_total = 0

            # Evaluate best model on test set
            for i, data in enumerate(sfgram_loader_test):
                # Data
                inputs, outputs = data
            # end for

            # Show loss
            print("Test loss {} ({})".format(
                test_loss / test_total,
                test_total
            ))

            # Print success rate
            xp.add_result(test_loss / test_total)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
