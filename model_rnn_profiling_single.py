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
import time

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

    # Feature transformer
    feature_transformer = features.create_transformer(
        feature,
        args.pretrained,
        args.embedding_path,
        lang
    )

    # Load PAN17 dataset
    pan17_dataset, pan17_dataset_per_tweet, pan17_loader_train, pan17_loader_dev, pan17_loader_test = dataset.load_pan17_dataset_per_tweet(
        output_length=settings.output_length[feature],
        output_dim=settings.input_dims[feature],
        batch_size=args.batch_size,
        trained=not args.pretrained,
        load_type=feature,
        transform=feature_transformer
    )

    # Print authors
    xp.write("Number of users : {}".format(len(pan17_dataset_per_tweet.user_tweets)), log_level=0)
    xp.write("Size of the dataset : {}".format(len(pan17_dataset_per_tweet)), log_level=0)
    xp.write("Number of users : {}".format(len(pan17_dataset.user_tweets)), log_level=0)
    xp.write("Size of the dataset : {}".format(len(pan17_dataset)), log_level=0)

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
            pan17_dataset_per_tweet.transform = features.create_transformer(
                feature,
                args.pretrained,
                args.embedding_path,
                lang
            )
            pan17_dataset.transform = pan17_dataset_per_tweet.transform

            # Model
            """rnn = rnn_func.create_profiling_model(
                feature=feature,
                pretrained=args.pretrained,
                cuda=args.cuda,
                embedding_dim=embedding_size,
                hidden_dim=hidden_size,
                vocab_size=settings.profiling_voc_size[feature],
                rnn_type=rnn_type,
                num_layers=num_layers,
                dropout=dropout,
                output_dropout=output_dropout,
                batch_size=args.batch_size,
                per_tweet=True
            )"""

            # Show model
            # print(rnn)

            # Optimizer
            # optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
            # optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0005)

            # Best model
            best_acc = 0.0
            best_model = None
            no_improv = 0

            # Loss function
            # loss_function = nn.MSELoss()
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
                # rnn.train()

                # Compute longest tweet
                longest_tweet = 0

                # Start time
                start = time.time()

                # Go through the training set
                for i, data in enumerate(pan17_loader_train):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data
                    """print("Inputs : {}".format(inputs.size()))
                    print("Gender : {}".format(gender))
                    print("Country : {}".format(country))
                    print("Gender vector : {}".format(gender_vector.size()))
                    print("Country vector : {}".format(country_vector))
                    print("Inputs lengths : {}".format(input_lengths))"""
                    if torch.max(input_lengths) > longest_tweet:
                        longest_tweet = torch.max(input_lengths)
                    # end if
                    print(torch.max(input_lengths))
                    # print("inputs : {}".format(inputs.size()))
                    # print(input_lengths)
                    # exit()
                    # Update tweet length
                    """if pan17_dataset.long_tweet > longest_tweet:
                        longest_tweet = pan17_dataset.long_tweet
                    # end if

                    # Lengths
                    batch_size = inputs.size(0)
                    tweets_size = inputs.size(1)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size)
                    for batch_i in range(batch_size):
                        indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                    # end for

                    # Transform to variable
                    inputs, indices_outputs = Variable(inputs), Variable(indices_outputs)

                    # To GPU
                    if use_cuda:
                        inputs, indices_outputs,  = inputs.cuda(), indices_outputs.cuda()
                        input_lengths = input_lengths.reshape(-1).cuda()
                    # end if

                    # Zero grad
                    rnn.zero_grad()

                    # Forward
                    model_outputs = rnn(inputs, input_lengths, reset_hidden=True)

                    # Average
                    model_outputs = torch.mean(model_outputs, dim=1)

                    # Class with highest probability
                    _, predicted_class = torch.max(model_outputs, dim=1)
                    
                    # Compute loss
                    loss = loss_function(model_outputs, indices_outputs)

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    # Add
                    training_acc += torch.sum(predicted_class == indices_outputs).item()
                    training_loss += loss.item()
                    training_total += batch_size
                    print(training_total)
                    if i > 10:
                        break
                    # end if"""
                # end for
                print(longest_tweet)
                # End time
                end = time.time()

                print(end - start)
                print(training_acc / training_total)
                print(training_total)
                exit()
                # Evaluation mode
                rnn.eval()

                # Go through the validation set
                for i, data in enumerate(pan17_loader_dev):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)
                    print("Batch size: {}".format(batch_size))
                    print("n_tweets: {}".format(batch_size))
                    print("tweets_size: {}".format(batch_size))
                    # For each profile
                    for profile_i in range(batch_size):
                        # Transform to variable
                        profile_inputs = inputs[profile_i]
                        profile_lengths = input_lengths[profile_i]
                        print("Profile inputs : {}".format(profile_inputs.size()))
                        print("Profile lengths : {}".format(profile_lengths.size()))
                        # Indices outputs
                        indices_outputs = torch.LongTensor(100).fill_(pan17_dataset.gender2num[gender[profile_i]])
                        print("Gender : {}".format(gender[profile_i]))
                        print("indices_outputs : {}".format(indices_outputs))
                        exit()
                        # Transform to variable
                        profile_inputs, profile_lengths = Variable(profile_inputs), Variable(profile_lengths)

                        # To GPU
                        if use_cuda:
                            profile_inputs = profile_inputs.cuda()
                            profile_lengths = profile_lengths.cuda()
                        # end if

                        # Forward
                        model_outputs = rnn(profile_inputs, profile_lengths, reset_hidden=True)

                        # Compute loss
                        loss = loss_function(model_outputs, indices_outputs)

                        # Average probability
                        output_prob = torch.mean(torch.mean(model_outputs, dim=1), dim=0)

                        # Predicted class
                        predicted_class = torch.argmax(output_prob)

                        # Add if ok
                        if predicted_class == pan17_dataset.gender2num[gender[profile_i]]:
                            validation_acc += 1
                        # end if

                        # Add
                        validation_loss += loss.item()
                        validation_total += 1
                    # end for
                # end for

                # Accuracies
                training_accuracy = training_acc / training_total * 100.0
                validation_accuracy = validation_acc / validation_total * 100.0

                # Keep best model
                if validation_accuracy > best_acc:
                    add_str = "##"
                    best_acc = validation_accuracy
                    torch.save(
                        rnn.state_dict(),
                        os.path.join(args.output, args.name, "rnn_profiling." + str(k) + u".pth")
                    )
                else:
                    add_str = ""
                # end if

                # Test loss
                test_loss = 0.0
                test_acc = 0.0
                test_total = 0

                # Go through the test set
                for i, data in enumerate(pan17_loader_test):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)

                    # For each profile
                    for profile_i in range(batch_size):
                        # Transform to variable
                        profile_inputs = inputs[profile_i]
                        profile_lengths = input_lengths[profile_i]

                        # Indices outputs
                        indices_outputs = torch.LongTensor(100).fill_(pan17_dataset.gender2num[gender[profile_i]])

                        # Transform to variable
                        profile_inputs, profile_lengths = Variable(profile_inputs), Variable(profile_lengths)

                        # To GPU
                        if use_cuda:
                            profile_inputs = profile_inputs.cuda()
                            profile_lengths = profile_lengths.cuda()
                        # end if

                        # Forward
                        model_outputs = rnn(profile_inputs, profile_lengths, reset_hidden=True)

                        # Compute loss
                        loss = loss_function(model_outputs, indices_outputs)

                        # Average probability
                        output_prob = torch.mean(torch.mean(model_outputs, dim=1), dim=0)

                        # Predicted class
                        predicted_class = torch.argmax(output_prob)

                        # Add if ok
                        if predicted_class == pan17_dataset.gender2num[gender[profile_i]]:
                            test_acc += 1
                        # end if

                        # Add
                        test_loss += loss.item()
                        test_total += 1
                    # end for
                # end for

                # Test accuracy
                test_accuracy = test_acc / test_total * 100.0

                # Show loss
                print("epoch {}, training loss {} ({}% / {}), validation loss {} ({}% / {}), test loss {} ({}%, {}) {}".format(
                    epoch,
                    training_loss / training_total,
                    round(training_accuracy, 2),
                    training_total,
                    validation_loss / validation_total,
                    round(validation_accuracy, 2),
                    validation_total,
                    test_loss / test_total,
                    round(test_accuracy, 2),
                    test_total,
                    add_str
                ))

                # Print longest tweet
                # print(longest_tweet)
            # end for

            # Test loss
            test_loss = 0.0
            test_acc = 0.0
            test_total = 0

            # Load best model
            rnn.load_state_dict(
                torch.load(
                    # open(os.path.join(args.output, args.name, u"rnn_profiling." + str(k) + u".pth"), 'rb')
                    os.path.join(args.output, args.name, u"rnn_profiling." + str(k) + u".pth")
                )
            )

            # Eval mode
            rnn.eval()

            # Go through the test set
            for i, data in enumerate(pan17_loader_test):
                # Data
                inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                # Lengths
                batch_size = inputs.size(0)
                n_tweets = inputs.size(1)
                tweets_size = inputs.size(2)

                # For each profile
                for profile_i in range(batch_size):
                    # Transform to variable
                    profile_inputs = inputs[profile_i]
                    profile_lengths = input_lengths[profile_i]

                    # Indices outputs
                    indices_outputs = torch.LongTensor(100).fill_(pan17_dataset.gender2num[gender[profile_i]])

                    # Transform to variable
                    profile_inputs, profile_lengths = Variable(profile_inputs), Variable(profile_lengths)

                    # To GPU
                    if use_cuda:
                        profile_inputs = profile_inputs.cuda()
                        profile_lengths = profile_lengths.cuda()
                    # end if

                    # Forward
                    model_outputs = rnn(profile_inputs, profile_lengths, reset_hidden=True)

                    # Compute loss
                    loss = loss_function(model_outputs, indices_outputs)

                    # Average probability
                    output_prob = torch.mean(torch.mean(model_outputs, dim=1), dim=0)

                    # Predicted class
                    predicted_class = torch.argmax(output_prob)

                    # Add if ok
                    if predicted_class == pan17_dataset.gender2num[gender[profile_i]]:
                        test_acc += 1
                    # end if

                    # Add
                    test_loss += loss.item()
                    test_total += 1
                # end for
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
            xp.add_result(test_accuracy)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
