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
from tools import argument_parsing, dataset, functions, settings
from tools import rnn as rnn_func
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os


# Evaluation pass
def evaluation_pass(sfgram_loader_set, thrs, rnn_model):
    """
    Evaluation pass
    :param sfgram_loader_set:
    :return:
    """
    # Test loss and total
    set_loss = 0.0
    set_total = 0.0

    # Test f1-scores
    set_true_positives = torch.zeros(n_threshold)
    set_false_positives = torch.zeros(n_threshold)
    set_false_negatives = torch.zeros(n_threshold)
    set_true_negatives = torch.zeros(n_threshold)

    # Set f1-scores
    set_f1_scores = torch.zeros(n_threshold)

    # Evaluate best model on test set
    for i, data in enumerate(sfgram_loader_set):
        # Data
        inputs, outputs, document_true = data

        # Transform to variable
        inputs, outputs = Variable(inputs), Variable(outputs)

        # To GPU
        if use_cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        # end if

        # Forward
        model_outputs = rnn_model(inputs, reset_hidden=True)

        # Compute loss
        loss = loss_function(model_outputs, outputs)

        # Test each threshold
        for threshold_i, threshold in enumerate(thresholds):
            # Threshold prediction
            if use_cuda:
                threshold_prediction = torch.zeros(model_outputs.size()).cuda()
            else:
                threshold_prediction = torch.zeros(model_outputs.size())
            # end if

            # Above threshold => true, false otherwise
            threshold_prediction[model_outputs > threshold] = 1.0
            threshold_prediction[model_outputs <= threshold] = 0.0

            # True positives
            true_index = threshold_prediction == 1.0
            false_index = threshold_prediction == 0.0

            # Add counts
            set_true_positives[threshold_i] += float((threshold_prediction[true_index] == outputs[true_index]).sum())
            set_false_positives[threshold_i] += float((threshold_prediction[true_index] != outputs[true_index]).sum())
            set_true_negatives[threshold_i] += float((threshold_prediction[false_index] == outputs[false_index]).sum())
            set_false_negatives[threshold_i] += float((threshold_prediction[false_index] != outputs[false_index]).sum())
        # end for

        # Add
        set_loss += loss.item()
        set_total += 1
    # end for

    # Compute F1-score for each threshold
    for threshold_i, threshold in enumerate(thrs):
        # Precision
        if set_true_positives[threshold_i] + set_false_positives[threshold_i] > 0:
            precision = set_true_positives[threshold_i] / (
                    set_true_positives[threshold_i] + set_false_positives[threshold_i])
        else:
            precision = 0.0
        # end if

        # Recall
        if set_true_positives[threshold_i] + set_false_negatives[threshold_i] > 0.0:
            recall = set_true_positives[threshold_i] / (
                    set_true_positives[threshold_i] + set_false_negatives[threshold_i])
        else:
            recall = 0.0
        # end if

        # Compute F1
        if precision > 0 and recall > 0:
            set_f1_scores[threshold_i] = 2.0 * (precision * recall) / (precision + recall)
        else:
            set_f1_scores[threshold_i] = 0.0
        # end if
    # end for

    return set_total, set_loss, set_f1_scores
# end evaluation_pass


#region INIT

# Param
use_mean_loss = 'arithmetic'
use_l1_loss = False
n_threshold = 20
thresholds = torch.linspace(0.0, 1.0, n_threshold)

# Seed
np.random.seed(1)
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
    sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test = dataset.load_sfgram_precomputed_dataset(
        block_length=40,
        batch_size=args.batch_size,
        author='ASIMOV',
        pretrained=args.pretrained,
        feature=feature,
        dev_per_file=True
    )

    # Print dataset information
    xp.write("TRAIN Dataset length : {}".format(sfgram_loader_train.dataset.data_length), log_level=0)
    xp.write("TRAIN Number of samples : {}".format(sfgram_loader_train.dataset.dataset_size), log_level=0)
    xp.write("TRAIN Number of trues : {}".format(sfgram_loader_train.dataset.trues_count), log_level=0)
    xp.write("DEV Dataset length : {}".format(sfgram_loader_dev.dataset.data_length), log_level=0)
    xp.write("DEV Number of samples : {}".format(sfgram_loader_dev.dataset.dataset_size), log_level=0)
    xp.write("DEV Number of trues : {}".format(sfgram_loader_dev.dataset.trues_count), log_level=0)
    xp.write("TEST Dataset length : {}".format(sfgram_loader_test.dataset.data_length), log_level=0)
    xp.write("TEST Number of samples : {}".format(sfgram_loader_test.dataset.dataset_size), log_level=0)
    xp.write("TEST Number of trues : {}".format(sfgram_loader_test.dataset.trues_count), log_level=0)

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
            sfgram_loader_dev.dataset.set_fold(k)
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
            optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)

            # Best model
            best_f1 = 0.0
            best_model = None

            # Loss function
            if not use_l1_loss:
                loss_function = nn.MSELoss()
            else:
                loss_function = nn.L1Loss()
            # end if

            # For each epoch
            for epoch in range(args.epoch):
                # Training, validation losses
                training_loss = 0.0
                training_total = 0.0
                training_f1_score = torch.zeros(n_threshold)

                # Number of samples
                n_samples = len(sfgram_loader_train)

                # RNN in training mode
                rnn.train()

                # Train f1-scores
                train_f1_score = torch.zeros(n_threshold)
                train_true_positives = torch.zeros(n_threshold)
                train_false_positives = torch.zeros(n_threshold)
                train_false_negatives = torch.zeros(n_threshold)
                train_true_negatives = torch.zeros(n_threshold)

                # Go through the training set
                for i, data in enumerate(sfgram_loader_train):
                    # Data
                    inputs, outputs, document_true = data

                    # Lengths
                    batch_size = inputs.size(0)
                    block_length = inputs.size(1)

                    # Transform to variable
                    inputs, outputs = Variable(inputs), Variable(outputs)

                    # To GPU
                    if use_cuda:
                        inputs, outputs = inputs.cuda(), outputs.cuda()
                    # end if

                    # Zero grad
                    rnn.zero_grad()

                    # Forward
                    model_outputs = rnn(inputs, reset_hidden=True)

                    if use_mean_loss == 'harmonic':
                        # Index of zero and one outputs
                        zero_index = (outputs == 0.0)
                        one_index = (outputs == 1.0)

                        # Loss for each class
                        loss_zero = loss_function(model_outputs[zero_index], outputs[zero_index])
                        loss_one = loss_function(model_outputs[one_index], outputs[one_index])

                        # Harmonic loss
                        loss = 2.0 * ((loss_zero * loss_one) / (loss_zero + loss_one))
                    elif use_mean_loss == 'arithmetic':
                        # Index of zero and one outputs
                        zero_index = (outputs == 0.0)
                        one_index = (outputs == 1.0)

                        # Loss for each class
                        loss_zero = loss_function(model_outputs[zero_index], outputs[zero_index])
                        loss_one = loss_function(model_outputs[one_index], outputs[one_index])

                        # Harmonic loss
                        loss = (args.zero_weight * loss_zero + loss_one) / (args.zero_weight + 1)
                    else:
                        # Compute loss
                        loss = loss_function(model_outputs, outputs)
                    # end if

                    # Backward pass
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    # Test each threshold
                    for threshold_i, threshold in enumerate(thresholds):
                        # Threshold prediction
                        if use_cuda:
                            threshold_prediction = torch.zeros(model_outputs.size()).cuda()
                        else:
                            threshold_prediction = torch.zeros(model_outputs.size())
                        # end if

                        # Above threshold => true, false otherwise
                        threshold_prediction[model_outputs > threshold] = 1.0
                        threshold_prediction[model_outputs <= threshold] = 0.0

                        # True positives
                        true_index = threshold_prediction == 1.0
                        false_index = threshold_prediction == 0.0

                        # Add counts
                        train_true_positives[threshold_i] += float((threshold_prediction[true_index] == outputs[true_index]).sum())
                        train_false_positives[threshold_i] += float((threshold_prediction[true_index] != outputs[true_index]).sum())
                        train_true_negatives[threshold_i] += float((threshold_prediction[false_index] == outputs[false_index]).sum())
                        train_false_negatives[threshold_i] += float((threshold_prediction[false_index] != outputs[false_index]).sum())
                    # end for

                    # Add
                    training_loss += loss.item()
                    training_total += 1
                # end for

                # Compute F1-score for each threshold
                for threshold_i, threshold in enumerate(thresholds):
                    # Precision
                    if train_true_positives[threshold_i] + train_false_positives[threshold_i] > 0:
                        precision = train_true_positives[threshold_i] / (train_true_positives[threshold_i] + train_false_positives[threshold_i])
                    else:
                        precision = 0.0
                    # end if

                    # Recall
                    if train_true_positives[threshold_i] + train_false_negatives[threshold_i] > 0.0:
                        recall = train_true_positives[threshold_i] / (train_true_positives[threshold_i] + train_false_negatives[threshold_i])
                    else:
                        recall = 0.0
                    # end if

                    if precision > 0 and recall > 0:
                        train_f1_score[threshold_i] = 2.0 * (precision * recall) / (precision + recall)
                    else:
                        train_f1_score[threshold_i] = 0.0
                    # end if
                # end for

                # Eval mode
                rnn.eval()

                # Evaluate dev and test
                dev_total, dev_loss, dev_f1_scores = evaluation_pass(sfgram_loader_dev, thresholds, rnn)
                test_total, test_loss, test_f1_scores = evaluation_pass(sfgram_loader_test, thresholds, rnn)

                # Final F1-score
                final_f1_scores = torch.zeros(n_threshold)

                # Compute F1-score for each threshold
                for threshold_i, threshold in enumerate(thresholds):
                    final_f1_scores[threshold_i] = (dev_f1_scores[threshold_i] + test_f1_scores[threshold_i]) / 2.0
                # end for

                # Max. train and test f1
                max_train_f1 = torch.max(train_f1_score).item()
                max_dev_f1 = torch.max(dev_f1_scores).item()
                max_test_f1 = torch.max(test_f1_scores).item()
                max_final_f1 = torch.max(final_f1_scores).item()

                # Improved performance
                if max_final_f1 > best_f1:
                    best_f1 = max_final_f1
                    add_str = "##"
                else:
                    add_str = ""
                # end if

                # Show loss
                print("epoch {}, training loss {} ({}), training F1 {}, dev loss {} ({}), dev F1 {}, test loss {} ({}), test F1 {}, final loss {} ({}), final F1 {} {}".format(
                    epoch,
                    round(training_loss / training_total, 4),
                    training_total,
                    round(max_train_f1, 4),
                    round(dev_loss / dev_total, 4),
                    dev_total,
                    round(max_dev_f1, 4),
                    round(test_loss / test_total, 4),
                    test_total,
                    round(max_test_f1, 4),
                    round((dev_loss + test_loss) / (dev_total + test_total), 4),
                    dev_total + test_total,
                    round(max_final_f1, 4),
                    add_str
                ))
            # end for

            # Show loss
            print("Final loss {} ({}), F1-score {}".format(
                round((dev_loss + test_loss) / (dev_total + test_total), 5),
                dev_total + test_total,
                round(max_final_f1, 5)
            ))

            # Print success rate
            xp.add_result(max_final_f1)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
