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
import os

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

    # Load PAN17 dataset
    pan17_dataset, pan17_loader_train, pan17_loader_dev, pan17_loader_test = dataset.load_pan17_dataset(
        output_length=settings.output_length[feature]*2,
        output_dim=settings.input_dims[feature],
        batch_size=1,
        trained=not args.pretrained,
        load_type=feature
    )

    # Print authors
    xp.write("Number of users : {}".format(len(pan17_dataset.user_tweets)), log_level=0)

    # Choose the right transformer
    pan17_dataset.transform = features.create_transformer(
        feature,
        args.pretrained,
        args.embedding_path,
        lang
    )

    # Set experience state
    xp.set_state(space)

    # For each sample
    for n in range(args.n_samples):
        # Set sample state
        xp.set_sample_state(n)

        # Average
        average_k_fold = np.array([])

        # For each fold
        for k in range(10):
            # Change fold state
            xp.set_fold_state(k)
            pan17_loader_train.dataset.set_fold(k)
            pan17_loader_dev.dataset.set_fold(k)
            pan17_loader_test.dataset.set_fold(k)

            # Choose the right transformer
            pan17_dataset.transform = features.create_transformer(
                feature,
                args.pretrained,
                args.embedding_path,
                lang
            )

            # Model
            rnn = rnn_func.create_esn_profiling_model(
                feature=feature,
                hidden_dim=hidden_size,
                connectivity=0.25,
                leaky_rate=args.leaky_rate,
                spectral_radius=args.spectral_radius
            )

            # Training, validation losses
            validation_total = 0.0
            validation_acc = 0.0
            test_acc = 0.0
            test_total = 0

            # Number of samples
            n_samples = len(pan17_loader_train)

            # RNN in training mode
            rnn.train()

            # Compute longest tweet
            longest_tweet = 0

            # Go through the training set
            for i, data in enumerate(pan17_loader_train):
                # Data
                inputs, gender, _, [gender_vector, _], input_lengths = data

                # Concat inputs
                concat_inputs = None

                # For each tweet
                for tweet_i in range(100):
                    # Add
                    if tweet_i == 0:
                        concat_inputs = inputs[0, tweet_i, :input_lengths[0, tweet_i]]
                    else:
                        concat_inputs = torch.cat((concat_inputs, inputs[0, tweet_i, :input_lengths[0, tweet_i]]))
                    # end if
                # end if

                # Add batch dim
                concat_inputs = concat_inputs.unsqueeze(0)

                # Outputs
                concat_outputs = torch.zeros(1, concat_inputs.size(1), 2)
                concat_outputs[0, :, pan17_dataset.gender2num[gender[0]]] = 1.0

                # Transform to variable
                concat_inputs, concat_outputs = Variable(concat_inputs), Variable(concat_outputs)

                # Forward
                model_outputs = rnn(concat_inputs, concat_outputs)
                break
            # end for

            # Finalize training
            rnn.finalize()

            # Go through the validation set
            for i, data in enumerate(pan17_loader_dev):
                # Data
                inputs, gender, _, [gender_vector, _], input_lengths = data

                # Concat inputs
                concat_inputs = None

                # For each tweet
                for tweet_i in range(100):
                    # Add
                    if tweet_i == 0:
                        concat_inputs = inputs[0, tweet_i, :input_lengths[0, tweet_i]]
                    else:
                        concat_inputs = torch.cat((concat_inputs, inputs[0, tweet_i, :input_lengths[0, tweet_i]]))
                    # end if
                # end if

                # Add batch dim
                concat_inputs = concat_inputs.unsqueeze(0)

                # Transform to variable
                concat_inputs = Variable(concat_inputs)

                # Forward
                model_outputs = rnn(concat_inputs)

                # Average over time
                model_outputs = torch.mean(model_outputs, dim=1)[0]

                # Class with highest probability
                predicted_class = torch.argmax(model_outputs)

                # Add
                if predicted_class == pan17_dataset.gender2num[gender[0]]:
                    validation_acc += 1.0
                # end if
                validation_total += 1
            # end for

            # Accuracies
            validation_accuracy = validation_acc / validation_total * 100.0

            # Evaluate best model on test set
            for i, data in enumerate(pan17_loader_test):
                # Data
                inputs, gender, _, [gender_vector, _], input_lengths = data

                # Concat inputs
                concat_inputs = None

                # For each tweet
                for tweet_i in range(100):
                    # Add
                    if tweet_i == 0:
                        concat_inputs = inputs[0, tweet_i, :input_lengths[0, tweet_i]]
                    else:
                        concat_inputs = torch.cat((concat_inputs, inputs[0, tweet_i, :input_lengths[0, tweet_i]]))
                    # end if
                # end if

                # Add batch dim
                concat_inputs = concat_inputs.unsqueeze(0)

                # Transform to variable
                concat_inputs = Variable(concat_inputs)

                # Forward
                model_outputs = rnn(concat_inputs)

                # Average over time
                model_outputs = torch.mean(model_outputs, dim=1)[0]

                # Class with highest probability
                predicted_class = torch.argmax(model_outputs)

                # Add
                if predicted_class == pan17_dataset.gender2num[gender[0]]:
                    test_acc += 1.0
                # end if
                test_total += 1
            # end for

            # Test accuracy
            test_accuracy = test_acc / test_total * 100.0

            # Print success rate
            xp.add_result((test_accuracy + validation_accuracy) / 2.0)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
