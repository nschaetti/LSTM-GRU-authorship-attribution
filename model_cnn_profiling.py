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
from torchlanguage import models
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
        output_length=settings.output_length[feature],
        output_dim=settings.input_dims[feature],
        batch_size=args.batch_size,
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
            model = models.CNNCTweet(
                text_length=settings.output_length[feature],
                vocab_size=settings.profiling_voc_size[feature],
                embedding_dim=embedding_size,
                out_channels=(args.num_channels, args.num_channels, args.num_channels),
                dropout=output_dropout
            )
            model = model.cuda()

            # Optimizer
            # optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            # Best model
            best_acc = 0.0
            best_model = None
            no_improv = 0

            # Loss function
            # loss_function = nn.NLLLoss()
            loss_function = nn.CrossEntropyLoss()

            # For each epoch
            for epoch in range(args.epoch):
                # Training, validation losses
                training_loss = 0.0
                training_loss_total = 0.0
                training_total = 0.0
                training_acc = 0.0
                validation_loss = 0.0
                validation_total = 0.0
                validation_acc = 0.0
                validation_profile_acc = 0.0
                validation_profile_total = 0.0

                # Number of samples
                n_samples = len(pan17_loader_train)

                # RNN in training mode
                model.train()

                # Compute longest tweet
                longest_tweet = 0
                # max_index = 0

                # Go through the training set
                for i, data in enumerate(pan17_loader_train):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                    # Sizes
                    batch_size, n_tweets, tweet_length = inputs.size()

                    # Reshape to have only tweets
                    inputs = inputs.reshape(batch_size * n_tweets, tweet_length)
                    gender_vector = gender_vector.reshape(batch_size * n_tweets, tweet_length)
                    input_lengths = input_lengths.reshape(batch_size * n_tweets)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size * n_tweets)
                    for batch_i in range(batch_size):
                        for tweet_i in range(n_tweets):
                            indices_outputs[batch_i * n_tweets + tweet_i] = pan17_dataset.gender2num[gender[batch_i]]
                        # end for
                    # end for

                    # Update tweet length
                    if torch.max(input_lengths) > longest_tweet:
                        longest_tweet = torch.max(input_lengths)
                    # end if

                    # Create a random permutation
                    rand_perm = torch.randperm(batch_size * n_tweets)
                    inputs = inputs[rand_perm]
                    gender_vector = gender_vector[rand_perm]
                    input_lengths = input_lengths[rand_perm]
                    indices_outputs = indices_outputs[rand_perm]

                    # Extract batches
                    for subbatch_i in range(0, batch_size * n_tweets, args.batch_size):
                        # Sub batch
                        inputs_subbatch = inputs[subbatch_i:subbatch_i+args.batch_size]
                        indices_outputs_subbatch = indices_outputs[subbatch_i:subbatch_i + args.batch_size]
                        gender_vector_subbatch = gender_vector[subbatch_i:subbatch_i + args.batch_size]
                        input_lengths_subbatch = input_lengths[subbatch_i:subbatch_i + args.batch_size]

                        # Transform to variable
                        inputs_subbatch = Variable(inputs_subbatch)
                        gender_vector_subbatch = Variable(gender_vector_subbatch)
                        indices_outputs_subbatch = Variable(indices_outputs_subbatch)

                        # To GPU
                        if use_cuda:
                            inputs_subbatch  = inputs_subbatch.cuda()
                            gender_vector_subbatch = gender_vector_subbatch.cuda()
                            indices_outputs_subbatch = indices_outputs_subbatch.cuda()
                            input_lengths_subbatch = input_lengths_subbatch.cuda()
                        # end if

                        # Zero grad
                        model.zero_grad()

                        # Forward
                        model_outputs = model(inputs_subbatch)
                        # print(indices_outputs)

                        # Class with highest probability
                        # _, predicted_class = torch.max(model_outputs, dim=1)
                        predicted_class = torch.argmax(model_outputs, dim=1)

                        # Compute loss
                        # loss = loss_function(model_outputs, indices_outputs)
                        loss = loss_function(model_outputs, indices_outputs_subbatch)

                        # Backward pass
                        loss.backward()

                        # Update weights
                        optimizer.step()

                        # Add
                        training_acc += torch.sum(predicted_class == indices_outputs_subbatch).item()
                        training_loss += loss.item()
                        training_loss_total += 1
                        training_total += inputs_subbatch.size(0)
                    # end for
                # end for

                # Evaluation mode
                model.eval()

                # Go through the validation set
                for i, data in enumerate(pan17_loader_dev):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size)
                    for batch_i in range(batch_size):
                        indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                    # end for

                    # For each profile
                    for user_i in range(batch_size):
                        # Index targets
                        user_indices_outputs = torch.LongTensor(n_tweets)
                        for tweet_i in range(n_tweets):
                            user_indices_outputs[tweet_i] = indices_outputs[user_i]
                        # end for

                        # Transform to variable
                        user_inputs = Variable(inputs[user_i])
                        user_class = Variable(indices_outputs[user_i])
                        user_tweet_lengths = input_lengths[user_i]
                        user_indices_outputs = Variable(user_indices_outputs)

                        # To GPU
                        if use_cuda:
                            user_inputs, user_class = user_inputs.cuda(), user_class.cuda()
                            user_tweet_lengths = user_tweet_lengths.cuda()
                            user_indices_outputs = user_indices_outputs.cuda()
                        # end if

                        # Forward
                        model_outputs = model(user_inputs)

                        # Predicted class for each tweet
                        predicted_class = torch.argmax(model_outputs, dim=1)

                        # Predicted class for the profile
                        predicted_class_profile = torch.argmax(torch.mean(model_outputs, dim=0))

                        # Compute loss
                        loss = loss_function(model_outputs, user_indices_outputs)

                        # Add
                        validation_loss += loss.item()
                        validation_acc += torch.sum(predicted_class == user_indices_outputs).item()
                        validation_total += n_tweets
                        if user_class == predicted_class_profile:
                            validation_profile_acc += 1
                        # end if
                        validation_profile_total += 1
                    # end for
                # end for

                # Accuracies
                training_accuracy = training_acc / training_total * 100.0
                validation_accuracy = validation_acc / validation_total * 100.0
                validation_profile_accuracy = validation_profile_acc / validation_profile_total * 100.0

                # Keep best model
                if validation_profile_accuracy > best_acc:
                    add_str = "##"
                    best_acc = validation_profile_accuracy
                    torch.save(
                        model.state_dict(),
                        # open(os.path.join(args.output, args.name, u"rnn_profiling." + str(k) + u".pth"), 'wb')
                        os.path.join(args.output, args.name, u"cnn_profiling." + str(k) + u".pth")
                    )
                else:
                    add_str = ""
                # end if

                # Test loss
                test_loss = 0.0
                test_acc = 0.0
                test_total = 0
                test_profile_acc = 0.0
                test_profile_total = 0.0

                # Evaluate best model on test set
                for i, data in enumerate(pan17_loader_test):
                    # Data
                    inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                    # Lengths
                    batch_size = inputs.size(0)
                    n_tweets = inputs.size(1)
                    tweets_size = inputs.size(2)

                    # Index targets
                    indices_outputs = torch.LongTensor(batch_size)
                    for batch_i in range(batch_size):
                        indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                    # end for

                    # For each profile
                    for user_i in range(batch_size):
                        # Index targets
                        user_indices_outputs = torch.LongTensor(n_tweets)
                        for tweet_i in range(n_tweets):
                            user_indices_outputs[tweet_i] = indices_outputs[user_i]
                        # end for

                        # Transform to variable
                        user_inputs = Variable(inputs[user_i])
                        user_class = Variable(indices_outputs[user_i])
                        user_tweet_lengths = input_lengths[user_i]
                        user_indices_outputs = Variable(user_indices_outputs)

                        # To GPU
                        if use_cuda:
                            user_inputs, user_class = user_inputs.cuda(), user_class.cuda()
                            user_tweet_lengths = user_tweet_lengths.cuda()
                            user_indices_outputs = user_indices_outputs.cuda()
                        # end if

                        # Forward
                        model_outputs = model(user_inputs)

                        # Predicted class for each tweet
                        predicted_class = torch.argmax(model_outputs, dim=1)

                        # Predicted class for the profile
                        predicted_class_profile = torch.argmax(torch.mean(model_outputs, dim=0))

                        # Compute loss
                        loss = loss_function(model_outputs, user_indices_outputs)

                        # Add
                        test_loss += loss.item()
                        test_acc += torch.sum(predicted_class == user_indices_outputs).item()
                        test_total += n_tweets
                        if user_class == predicted_class_profile:
                            test_profile_acc += 1
                        # end if
                        test_profile_total += 1
                    # end for
                # end for

                # Test accuracy
                test_accuracy = test_acc / test_total * 100.0
                test_profile_accuracy = test_profile_acc / test_profile_total * 100.0

                # Show loss
                print("epoch {}, training loss {} ({}% / {}), validation loss {} ({}% / {}, {}% / {}), test loss {} ({}%, {}, {}% / {}) {}".format(
                    epoch,
                    round(training_loss / training_loss_total, 5),
                    round(training_accuracy, 2),
                    training_total,
                    round(validation_loss / validation_profile_total, 5),
                    round(validation_accuracy, 2),
                    validation_total,
                    round(validation_profile_accuracy, 2),
                    validation_profile_total,
                    round(test_loss / test_profile_total, 5),
                    round(test_accuracy, 2),
                    test_total,
                    round(test_profile_accuracy, 2),
                    test_profile_total,
                    add_str
                ))

                # Print longest tweet
                # print(longest_tweet)
            # end for

            # Test loss
            test_loss = 0.0
            test_acc = 0.0
            test_total = 0
            test_profile_acc = 0.0
            test_profile_total = 0.0

            # Load best model
            model.load_state_dict(
                torch.load(
                    # open(os.path.join(args.output, args.name, u"rnn_profiling." + str(k) + u".pth"), 'rb')
                    os.path.join(args.output, args.name, "cnn_profiling." + str(k) + u".pth")
                )
            )

            # Eval mode
            model.eval()

            # Evaluate best model on test set
            for i, data in enumerate(pan17_loader_test):
                # Data
                inputs, gender, country, [gender_vector, country_vector], input_lengths = data

                # Lengths
                batch_size = inputs.size(0)
                n_tweets = inputs.size(1)
                tweets_size = inputs.size(2)

                # Index targets
                indices_outputs = torch.LongTensor(batch_size)
                for batch_i in range(batch_size):
                    indices_outputs[batch_i] = pan17_dataset.gender2num[gender[batch_i]]
                # end for

                # For each profile
                for user_i in range(batch_size):
                    # Index targets
                    user_indices_outputs = torch.LongTensor(n_tweets)
                    for tweet_i in range(n_tweets):
                        user_indices_outputs[tweet_i] = indices_outputs[user_i]
                    # end for

                    # Transform to variable
                    user_inputs = Variable(inputs[user_i])
                    user_class = Variable(indices_outputs[user_i])
                    user_tweet_lengths = input_lengths[user_i]
                    user_indices_outputs = Variable(user_indices_outputs)

                    # To GPU
                    if use_cuda:
                        user_inputs, user_class = user_inputs.cuda(), user_class.cuda()
                        user_tweet_lengths = user_tweet_lengths.cuda()
                        user_indices_outputs = user_indices_outputs.cuda()
                    # end if

                    # Forward
                    model_outputs = model(user_inputs)

                    # Predicted class for each tweet
                    predicted_class = torch.argmax(model_outputs, dim=1)

                    # Predicted class for the profile
                    predicted_class_profile = torch.argmax(torch.mean(model_outputs, dim=0))

                    # Compute loss
                    loss = loss_function(model_outputs, user_indices_outputs)

                    # Add
                    test_loss += loss.item()
                    test_acc += torch.sum(predicted_class == user_indices_outputs).item()
                    test_total += n_tweets
                    if user_class == predicted_class_profile:
                        test_profile_acc += 1
                    # end if
                    test_profile_total += 1
                # end for
            # end for

            # Test accuracy
            test_accuracy = test_acc / test_total * 100.0
            test_profile_accuracy = test_profile_acc / test_profile_total * 100.0

            # Show loss
            print("Test loss {} ({}% / {})".format(
                test_loss / test_total,
                round(test_profile_accuracy, 2),
                test_profile_total
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
