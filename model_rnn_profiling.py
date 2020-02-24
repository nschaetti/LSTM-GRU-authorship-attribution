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

# Load PAN17 dataset
pan17_dataset, pan17_loader_train, pan17_loader_dev, pan17_loader_test = dataset.load_pan17_dataset(
    output_length=settings.c1_output_length,
    output_dim=settings.c1_output_dim,
    batch_size=args.batch_size
)

# Print authors
xp.write("Number of users : {}".format(len(pan17_dataset.user_tweets)), log_level=0)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, \
    num_layers, dropout, output_dropout = functions.get_params(space)

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
            pan17_loader_test.dataset.set_fold(k)

            # Choose the right transformer
            pan17_dataset.transform = features.create_transformer(
                feature,
                args.pretrained,
                args.embedding_path,
                lang
            )

            # Model
            rnn = rnn_func.create_profiling_model(
                feature=feature,
                pretrained=args.pretrained,
                cuda=args.cuda,
                embedding_dim=embedding_size,
                hidden_dim=hidden_size,
                vocab_size=settings.voc_size[feature],
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
            best_acc = -1.0
            best_model = None
            no_improv = 0

            # Loss function
            # loss_function = nn.NLLLoss()
            loss_function = nn.CrossEntropyLoss()

            # For each epoch
            for epoch in range(args.epoch):
                # Training, validation losses
                training_loss = 0.0
                training_total = 0.0
                training_acc = 0.0
                validation_loss = 0.0
                validation_total = 0.0
                validation_acc = 0.0

                # Number of samples
                n_samples = len(pan17_loader_train)

                # RNN in training mode
                rnn.train()

                # Compute longest tweet
                # longest_tweet = 0

                # Go through the training set
                for i, data in enumerate(pan17_loader_train):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector] = data

                    # Update tweet length
                    """if pan17_dataset.long_tweet > longest_tweet:
                        longest_tweet = pan17_dataset.long_tweet
                    # end if"""

                    # Transform to variable
                    # inputs, gender_vector = Variable(inputs), Variable(gender_vector)
                    inputs = Variable(inputs)

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size)
                    for batch_i in range(batch_size):
                        indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                    # end for

                    # Zero grad
                    rnn.zero_grad()

                    # Forward
                    model_outputs = rnn(inputs, reset_hidden=True)
                    # print(model_outputs)
                    # print(indices_outputs)

                    # Class with highest probability
                    _, predicted_class = torch.max(model_outputs, dim=1)

                    # Compute loss
                    loss = loss_function(model_outputs, indices_outputs)

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()
                    # print("Loss: {}".format(loss.item()))
                    # print("#", end='')
                    # Add
                    training_acc += torch.sum(predicted_class == indices_outputs).item() / float(batch_size)
                    training_loss += loss.item()
                    training_total += 1
                # end for

                # Evaluation mode
                rnn.eval()

                # Go through the validation set
                for i, data in enumerate(pan17_loader_dev):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector] = data

                    # Transform to variable
                    inputs = Variable(inputs)

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size)
                    for batch_i in range(batch_size):
                        indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                    # end for

                    # Forward
                    model_outputs = rnn(inputs, reset_hidden=True)

                    # Class with highest probability
                    _, predicted_class = torch.max(model_outputs, dim=1)

                    # Compute loss
                    loss = loss_function(model_outputs, indices_outputs)

                    # Add
                    validation_loss += loss.item()
                    validation_acc += torch.sum(predicted_class == indices_outputs).item() / float(batch_size)
                    validation_total += 1
                # end for

                # Accuracies
                training_accuracy = training_acc / training_total * 100.0
                validation_accuracy = validation_acc / validation_total * 100.0

                # Show loss
                print("epoch {}, training loss {} ({}% / {}), validation loss {} ({}% / {})".format(
                    epoch,
                    training_loss / training_total,
                    round(training_accuracy, 2),
                    training_total,
                    validation_loss / validation_total,
                    round(validation_accuracy, 2),
                    validation_total
                ))

                # Keep best model
                if validation_accuracy > best_acc:
                    best_acc = validation_accuracy
                    best_model = rnn
                # end if

                # Print longest tweet
                # print(longest_tweet)
            # end for

            # Test loss
            test_loss = 0.0
            test_acc = 0.0
            test_total = 0

            # Evaluate best model on test set
            for i, data in enumerate(pan17_loader_test):
                # Data
                inputs, gender, country, [gender_vector, country_vector] = data

                # Transform to variable
                inputs = Variable(inputs)

                # Lengths
                batch_size = inputs.size(0)
                n_tweets = inputs.size(1)
                tweets_size = inputs.size(2)

                # Index targets
                indices_outputs = torch.LongTensor(batch_size)
                for batch_i in range(batch_size):
                    indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                # end for

                # Forward
                model_outputs = best_model(inputs, reset_hidden=True)

                # Class with highest probability
                _, predicted_class = torch.max(model_outputs, dim=1)

                # Compute loss
                loss = loss_function(model_outputs, indices_outputs)

                # Loss
                test_loss += loss.item()
                test_acc += torch.sum(predicted_class == indices_outputs).item() / float(batch_size)
                test_total += 1
            # end for

            # Test accuracy
            test_accuracy = test_acc / test_total * 100.0

            # Show loss
            print("Test loss {} ({}% / {})".format(
                test_loss / test_total,
                round(test_accuracy, 2),
                test_total
            ))

            # Print success rate
            xp.add_result(test_acc / test_total)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()