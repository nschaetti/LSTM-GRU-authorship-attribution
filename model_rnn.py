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
from tools import argument_parsing, dataset, functions, features, settings
from tools import rnn as rnn_func
from torch import optim
import torch.nn as nn
import os


# Test a model
def test_model(reuters_load_set):
    """
    Test a model
    :param reuters_load_set:
    :return:
    """
    # Counters
    success = 0.0
    count = 0.0
    test_loss = 0.0
    test_total = 0.0

    # Get test data for this fold
    for i, data in enumerate(reuters_load_set):
        # Inputs and labels
        inputs, labels, _ = data

        # Time labels
        time_labels = torch.LongTensor(1, inputs.size(1)).fill_(labels[0])

        # Sequences length
        input_length = inputs.size(1)
        print(i)
        # Batch
        if i % args.test_batch_size == 0 or i == 149:
            if i != 0:
                if i == 149:
                    batch_list.append((inputs, time_labels, input_length, labels))
                # end if

                # Sort list
                batch_list.sort(key=lambda tup: tup[2], reverse=True)

                # Transformation to tensors
                batch_tensor, batch_labels, batch_lengths, batch_y = functions.list_to_tensors(batch_list)

                # Variable and CUDA
                batch_tensor, batch_labels, batch_lengths, batch_y = Variable(batch_tensor), Variable(
                    batch_labels), Variable(batch_lengths), Variable(batch_y)
                if args.cuda:
                    batch_tensor, batch_labels, batch_lengths, batch_y = batch_tensor.cuda(), batch_labels.cuda(), batch_lengths.cuda(), batch_y.cuda()
                # end if

                # Forward
                model_outputs = rnn(batch_tensor, batch_lengths, reset_hidden=True)

                # Loss summation
                loss = 0

                # Loss for each sample
                for j in range(model_outputs.size(0)):
                    # Seq. length
                    seq_length = batch_lengths[j]
                    loss += loss_function(model_outputs[j, :seq_length],
                                          batch_labels[j, :seq_length])
                    _, pred = torch.max(torch.mean(model_outputs[j, :seq_length], dim=0), dim=0)
                    if pred.item() == batch_y[j].item():
                        success += 1.0
                    # end if
                    count += 1.0
                # end for

                # Add loss
                test_loss += loss.item()
                test_total += len(batch_list)
            # end if

            # Create list
            batch_list = list()

            # Add to list
            batch_list.append((inputs, time_labels, input_length, labels))
        else:
            # Add to list
            batch_list.append((inputs, time_labels, input_length, labels))
        # end if
    # end for

    # Compute accuracy
    accuracy = success / count * 100.0

    return accuracy, test_loss, test_total
# end test_model


####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_dev, reuters_loader_test = dataset.load_dataset(args.dataset_size)

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout = functions.get_params(space)

    # Choose the right transformer
    reutersc50_dataset.transform = features.create_transformer(
        feature,
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
                args.pretrained,
                args.embedding_path,
                lang
            )

            # Model
            rnn = rnn_func.create_model(
                feature=feature,
                pretrained=args.pretrained,
                cuda=args.cuda,
                embedding_dim=embedding_size,
                hidden_dim=hidden_size,
                vocab_size=settings.voc_size[feature],
                rnn_type=rnn_type,
                num_layers=num_layers,
                dropout=dropout,
                batch_size=args.batch_size
            )

            # Optimizer
            # optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.9)
            optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0001)

            # Best model
            best_acc = 0.0
            no_improv = 0

            # Loss function
            loss_function = nn.NLLLoss()

            # For each epoch
            for epoch in range(args.epoch):
                # Total losses
                training_loss = 0.0
                training_total = 0.0

                # Training
                rnn.train()

                # Get training data for this fold
                for i, data in enumerate(reuters_loader_train):
                    # Inputs and labels
                    inputs, labels, _ = data
                    print(i)
                    # Time labels
                    time_labels = torch.LongTensor(1, inputs.size(1)).fill_(labels[0])

                    # Input length
                    input_length = inputs.size(1)

                    # Batch
                    if i % args.batch_size == 0 or i == 1159:
                        if i != 0:
                            if i == 1159:
                                batch_list.append((inputs, time_labels, input_length, labels))
                            # end if

                            # Sort list
                            batch_list.sort(key=lambda tup: tup[2], reverse=True)

                            # Transformation to tensors
                            batch_tensor, batch_labels, batch_lengths, batch_y = functions.list_to_tensors(batch_list)

                            # Loss summation
                            loss = 0
                            sequence_count = 0.0

                            # Max sequence length
                            for j in range(0, batch_tensor.size(1), args.max_length):
                                # Sub. sequence
                                sequence_tensor = batch_tensor[:, j:j+args.max_length]
                                sequence_labels = batch_labels[:, j:j+args.max_length]
                                sequence_length = batch_lengths.clone()
                                sequence_length[sequence_length.gt(args.max_length)] = args.max_length

                                # Select only sequence length > 0
                                sequence_mask = sequence_length.gt(0)
                                sequence_tensor = sequence_tensor[sequence_mask]
                                sequence_labels = sequence_labels[sequence_mask]
                                sequence_length = sequence_length[sequence_mask]

                                # Variable and CUDA
                                sequence_tensor, sequence_labels, sequence_length = Variable(sequence_tensor), Variable(sequence_labels), Variable(sequence_length)
                                if args.cuda:
                                    sequence_tensor, sequence_labels, sequence_length = sequence_tensor.cuda(), sequence_labels.cuda(), sequence_length.cuda()
                                # end if

                                # Zero grad
                                rnn.zero_grad()
                                # print(sequence_tensor.shape)
                                # Forward
                                if j == 0:
                                    model_outputs = rnn(sequence_tensor, sequence_length, sequence_mask, reset_hidden=True)
                                else:
                                    model_outputs = rnn(sequence_tensor, sequence_length, sequence_mask, reset_hidden=False)
                                # end if

                                # Loss summation
                                # loss = 0

                                # Loss for each sample
                                for k in range(model_outputs.size(0)):
                                    # Seq. length
                                    seq_length = sequence_length[k]
                                    loss += loss_function(model_outputs[k, :seq_length],
                                                          sequence_labels[k, :seq_length])
                                    sequence_count += 1.0
                                # end for
                                """print(u"Batch {}, Sequence {}, loss {}".format(i, j, loss))
                                # Backward and step
                                if i < batch_tensor.size(1) - 1:
                                    loss.backward(retain_graph=True)
                                else:
                                    loss.backward()
                                # end if

                                # Step
                                optimizer.step()

                                # Add
                                training_loss += loss.item()
                                training_total += sequence_tensor.size(0)"""

                                # Remove sequence length
                                batch_lengths -= args.max_length
                                batch_lengths[batch_lengths.lt(0)] = 0
                            # end if
                            # print(u"Batch {}, loss {} ({})".format(i, loss / sequence_count, sequence_count))
                            # Backward and step
                            loss.backward()
                            optimizer.step()

                            # Add
                            training_loss += loss.item()
                            training_total += sequence_count
                        # end if

                        # Create list
                        batch_list = list()

                        # Add to list
                        batch_list.append((inputs, time_labels, input_length, labels))
                    else:
                        # Add to list
                        batch_list.append((inputs, time_labels, input_length, labels))
                    # end if
                # end for

                # Counters
                success = 0.0
                count = 0.0

                # Evaluation
                rnn.eval()
    
                # Test model
                dev_accuracy, dev_loss, dev_total = test_model(reuters_loader_dev)

                # Print
                xp.write(u"Epoch {}, training loss {} ({}), dev loss {} ({}), dev accuracy {}".format(
                    epoch,
                    training_loss / training_total,
                    training_total,
                    dev_loss / dev_total,
                    dev_total,
                    dev_accuracy
                ), log_level=5)

                # Save if best
                if dev_accuracy > best_acc:
                    best_acc = dev_accuracy
                    torch.save(
                        rnn.state_dict(),
                        open(os.path.join(args.output, args.name, u"rnn." + str(k) + u".pth"), 'wb')
                    )
                    if not args.pretrained:
                        torch.save(
                            reutersc50_dataset.transform.transforms[1].token_to_ix,
                            open(os.path.join(args.output, args.name, u"rnn." + str(k) + u".voc.pth"), 'wb')
                        )
                    # end if
                # end if
            # end for

            # Now we test on the test set
            test_accuracy, test_loss, test_total = test_model(reuters_loader_dev)

            # Print success rate
            xp.add_result(test_loss / test_total)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()
