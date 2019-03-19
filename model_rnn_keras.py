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
from tools import argument_parsing, dataset, functions, features, settings
from tools import keras_func as keras_tools
from models import KerasRNN
from keras import optimizers
import keras
import math

####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_dev, reuters_loader_test = dataset.load_dataset(
    args.dataset_size)

# Print authors
xp.write(u"Authors : {}".format(reutersc50_dataset.authors), log_level=0)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout, output_dropout = functions.get_params(
        space)

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
    certainty_data = np.zeros((2, args.n_samples * len(reutersc50_dataset.authors) * 100))
    certainty_index = 0

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # For each fold
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

            # Create the model
            model = KerasRNN.create_rnn_model(
                rnn_type=rnn_type,
                voc_size=settings.voc_size[feature],
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                dense_size=args.n_authors,
                average=False
            )

            # Print model summary
            print(model.summary(90))

            # Adam
            adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            # Get train, dev and test as lists
            train_inputs, train_labels, train_time_labels = keras_tools.dataset_to_list(reuters_loader_train, settings.voc_size[feature])
            dev_inputs, dev_labels, dev_time_labels = keras_tools.dataset_to_list(reuters_loader_dev, settings.voc_size[feature])
            test_inputs, test_labels, test_time_labels = keras_tools.dataset_to_list(reuters_loader_test, settings.voc_size[feature])

            # Training generator
            training_generator = keras_tools.ReutersC50BatchGenerator(
                data_inputs=train_inputs,
                data_labels=train_time_labels,
                # data_labels=train_labels,
                batch_size=args.batch_size,
                num_classes=args.n_authors,
                many_to_many=True,
                max_index=1000000
            )

            # Validation generator
            validation_generator = keras_tools.ReutersC50BatchGenerator(
                data_inputs=dev_inputs,
                data_labels=dev_time_labels,
                # data_labels=dev_labels,
                batch_size=args.batch_size,
                num_classes=args.n_authors,
                many_to_many=True,
                max_index=1000000
            )

            # For each epoch
            for epoch in range(args.epoch):
                # Train
                model.fit_generator(
                    generator=training_generator.generate(),
                    steps_per_epoch=math.ceil(80.0 * args.n_authors / args.batch_size),
                    epochs=args.epoch,
                    verbose=1,
                    validation_data=validation_generator.generate(),
                    validation_steps=math.ceil(10.0 * args.n_authors / args.batch_size),
                    use_multiprocessing=False,
                    workers=0
                )
            # end for

            # Print success rate
            xp.add_result(0.0)
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()

