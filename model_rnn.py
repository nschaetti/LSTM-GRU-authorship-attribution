#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
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
from torch.autograd import Variable
import echotorch.nn as etnn
import echotorch.utils
from tools import argument_parsing, dataset, functions, features
import matplotlib.pyplot as plt


####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_test = dataset.load_dataset(args.dataset_size)

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window = functions.get_params(space)

    # Choose the right transformer
    reutersc50_dataset.transform = features.create_transformer(
        feature,
        learning_window,
        args.pretrained,
        args.embedding_path,
        lang
    )

    # Dataset start
    reutersc50_dataset.set_start(dataset_start)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # Certainty data
    certainty_data = np.zeros((2, args.n_samples * 1500))
    certainty_index = 0

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Model

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # For each batch
        for k in range(10):
            # Choose fold
            xp.set_fold_state(k)
            reuters_loader_train.dataset.set_fold(k)
            reuters_loader_test.dataset.set_fold(k)

            # Choose the right transformer
            reutersc50_dataset.transform = features.create_transformer(
                feature,
                learning_window,
                args.pretrained,
                args.embedding_path, lang
            )

            # Get training data for this fold
            for i, data in enumerate(reuters_loader_train):
                # Inputs and labels
                inputs, labels, time_labels = data

                # Reset hidden state

                # For each training window
                for j in range(inputs.size(1)):
                    # To variable
                    window_inputs, window_time_labels = Variable(inputs[0, j]), Variable(time_labels[0, j])
                    if use_cuda: window_inputs, window_time_labels = window_inputs.cuda(), window_time_labels.cuda()

                    # TRAINING
                    print(window_inputs)
                    print(labels)
                    print(window_time_labels)
                    if j == 10:
                        exit()
                    # end if
                # end for
            # end for

            # Counters
            successes = 0.0
            count = 0.0
            local_success = 0.0
            local_count = 0.0

            # Get test data for this fold
            for i, data in enumerate(reuters_loader_test):
                # Inputs and labels
                inputs, labels, time_labels = data

                # Time labels
                local_labels = torch.LongTensor(1, time_labels.size(1)).fill_(labels[0])

                # To variable
                inputs, labels, time_labels, local_labels = Variable(inputs), Variable(labels), Variable(time_labels), Variable(local_labels)
                if use_cuda: inputs, labels, time_labels, local_labels = inputs.cuda(), labels.cuda(), time_labels.cuda(), local_labels.cuda()

                # TESTING
            # end for

            # Compute accuracy
            if args.measure == 'global':
                accuracy = successes / count
            else:
                accuracy = local_success / local_count
            # end if

            # Print success rate
            xp.add_result(accuracy)
        # end for
    # end for

    # Save certainty
    if args.certainty != "":
        print(certainty_data)
        np.save(open(args.certainty, "wb"), certainty_data)
    # end if

    # W index
    w_index += 1

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
