#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : model_rnn_keras.py
# Description : Test RNN models with Keras.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the RNN-authorship-attribution Project.
# The RNN-authorship-attribution Project is a set of free software:
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
# along with RNN-authorship-attribution.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
from tools import argument_parsing, dataset, functions, features, settings
from tools import keras_func as keras_tools
from models import KerasRNN
import generators as G
from keras import optimizers
from keras import callbacks
import math
from tools import load_glove_embeddings as gle
import os

####################################################
# Main
####################################################


# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Load from directory
if args.inverse_dev_test:
    pan17_dataset, pan17_loader_train, pan17_loader_dev, pan17_loader_test = dataset.load_pan17_dataset(
        k=args.k
    )
else:
    pan17_dataset, pan17_loader_train, pan17_loader_test, pan17_loader_dev = dataset.load_dataset(
        args.dataset_size,
        k=args.k,
        n_authors=args.n_authors
    )
# end if

# Disable CUDA
if not args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# end if

# Print authors
xp.write(u"Twitter users : {}".format(len(pan17_dataset)), log_level=0)
dataset_size = len(pan17_dataset)

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout, output_dropout = functions.get_params(
        space)

    # Set experience state
    xp.set_state(space)

    # Average sample
    average_sample = np.array([])

    # Load GloVe if needed
    if args.pretrained and args.fine_tuning:
        word2index, embedding_matrix, pretrained_vocsize = features.load_pretrained_weights(
            feature=feature,
            emb_path=args.embedding_path,
            embedding_size=embedding_size
        )
    else:
        word2index = None
        embedding_matrix = None
        pretrained_vocsize = 0
    # end if

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # For each fold
        for k in range(args.k):
            # Choose fold
            xp.set_fold_state(k)
            pan17_loader_train.dataset.set_fold(k)
            pan17_loader_dev.dataset.set_fold(k)
            pan17_loader_test.dataset.set_fold(k)

            # Choose the right transformer
            pan17_dataset.transform = features.create_transformer(
                feature,
                True if args.pretrained and not args.fine_tuning else False,
                args.embedding_path,
                lang,
                token2index=word2index
            )

            # Create the model
            if args.pretrained and not args.fine_tuning:
                # Create model
                model = KerasRNN.create_rnn_model(
                    rnn_type=rnn_type,
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                    dense_size=2,
                    average=False
                )
            elif args.pretrained and args.fine_tuning:
                # Create model
                model = KerasRNN.create_rnn_model_with_pretrained_embedding_layer(
                    rnn_type=rnn_type,
                    embedding_matrix=embedding_matrix,
                    hidden_size=hidden_size,
                    dense_size=2,
                    average=False,
                    trainable=True
                )
            else:
                # Create model
                model = KerasRNN.create_rnn_model_with_embedding_layer(
                    rnn_type=rnn_type,
                    voc_size=settings.voc_size[feature],
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                    dense_size=2,
                    average=False
                )
            # end if

            # Print model summary
            if k == 0:
                print(model.summary(90))
            # end if

            # Adam
            adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

            # Get train, dev and test as lists
            if args.pretrained and not args.fine_tuning:
                train_inputs, train_labels, train_time_labels = keras_tools.dataset_to_numpy(reuters_loader_train)
                dev_inputs, dev_labels, dev_time_labels = keras_tools.dataset_to_numpy(reuters_loader_dev)
                test_inputs, test_labels, test_time_labels = keras_tools.dataset_to_numpy(reuters_loader_test)
            else:
                train_inputs, train_labels, train_time_labels = keras_tools.dataset_to_list(reuters_loader_train, settings.voc_size[feature])
                dev_inputs, dev_labels, dev_time_labels = keras_tools.dataset_to_list(reuters_loader_dev, settings.voc_size[feature])
                test_inputs, test_labels, test_time_labels = keras_tools.dataset_to_list(reuters_loader_test, settings.voc_size[feature])
            # end if

            # Training generator
            training_generator = G.ReutersC50BatchGenerator(
                data_inputs=train_inputs,
                data_labels=train_time_labels,
                batch_size=args.batch_size,
                num_classes=args.n_authors,
                many_to_many=True,
                max_index=pretrained_vocsize if args.fine_tuning else -1,
                pretrained=True
            )

            # Validation generator
            validation_generator = G.ReutersC50BatchGenerator(
                data_inputs=dev_inputs,
                data_labels=dev_time_labels,
                batch_size=args.batch_size,
                num_classes=args.n_authors,
                many_to_many=True,
                max_index=pretrained_vocsize if args.fine_tuning else -1,
                pretrained=True
            )

            # Test generator
            test_generator = G.ReutersC50BatchGenerator(
                data_inputs=test_inputs,
                data_labels=test_time_labels,
                batch_size=args.batch_size,
                num_classes=args.n_authors,
                many_to_many=True,
                max_index=pretrained_vocsize if args.fine_tuning else -1,
                pretrained=True
            )

            # Model checkpoint
            checkpoint = callbacks.ModelCheckpoint(
                "saved_models/model_{}_keras-{}-{}-{}-{}-{}-{}.h5".format(rnn_type, feature, hidden_size, embedding_size, num_layers, args.n_authors, k),
                verbose=1,
                monitor='val_loss',
                save_best_only=True,
                mode='auto'
            )

            # Generators
            if args.pretrained and not args.fine_tuning:
                train_generate = training_generator.generate_embeddings()
                validation_generate = validation_generator.generate_embeddings()
                test_generate = test_generator.generate_embeddings()
            else:
                train_generate = training_generator.generate_indexes()
                validation_generate = validation_generator.generate_indexes()
                test_generate = test_generator.generate_indexes()
            # end if

            # Train and validation
            model.fit_generator(
                generator=train_generate,
                steps_per_epoch=math.ceil(90.0 * dataset_size / args.batch_size),
                epochs=args.epoch,
                verbose=1,
                validation_data=validation_generate,
                validation_steps=math.ceil(10.0 * dataset_size / args.batch_size),
                use_multiprocessing=False,
                workers=0,
                callbacks=[checkpoint]
            )

            # Load best model
            model.load_weights("saved_models/model_{}_keras-{}-{}-{}-{}-{}-{}.h5".format(rnn_type, feature, hidden_size, embedding_size, num_layers, args.n_authors, k))

            # Counters
            count = 0.0
            total = 0.0

            # Test
            for i, batch in enumerate(test_generate):
                # Inputs and outputs
                x_test, y_test = batch

                # Predict
                predictions = model.predict(x_test, batch_size=args.batch_size)

                # Average author probabilities
                predictions = np.average(predictions, axis=1)

                # Maximum probabilities
                predicted = np.argmax(predictions, axis=1)

                # Ground truth
                truth = np.argmax(np.average(y_test, axis=1), axis=1)

                # Correctly predicted
                count += np.sum(predicted == truth)

                # Total
                total += y_test.shape[0]

                # End ?
                if total >= 360:
                    break
                # end if
            # end for

            # Accuracy
            accuracy = count / total

            # Print success rate
            xp.add_result(accuracy)

            # Delete
            del model
            del training_generator
            del validation_generator
            del test_generator
            del train_inputs
            del train_labels
            del train_time_labels
        # end for
    # end for

    # Last space
    last_space = space
# end for

# Save experiment results
xp.save()

