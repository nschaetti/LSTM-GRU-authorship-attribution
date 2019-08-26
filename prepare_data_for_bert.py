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
import matplotlib.pyplot as plt
import argparse
import torchlanguage.transforms
import codecs

####################################################
# Main
####################################################

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()

# Load from directory
reutersc50_dataset, reuters_loader_train, reuters_loader_dev, reuters_loader_test = dataset.load_dataset(
    k=args.datadir,
    n_authors=15
)

# Tokenizer transformer
reutersc50_dataset.transform = torchlanguage.transforms.Token()

# For each fold
for k in range(args.datadir):
    # Fold paths
    fold_dir = os.path.join(args.datadir, u"k{}".format(k))
    fold_train_dir = os.path.join(fold_dir, u"train")
    fold_test_dir = os.path.join(fold_dir, u"test")

    # Create directories
    os.mkdir(fold_dir)
    os.mkdir(fold_train_dir)
    os.mkdir(fold_test_dir)

    # Create authors directories
    for a in range(15):
        # Author path
        author_train_dir = os.path.join(fold_train_dir, u"class{}".format(a))
        author_test_dir = os.path.join(fold_test_dir, u"class{}".format(a))

        # Create
        os.mkdir(author_train_dir)
        os.mkdir(author_test_dir)
    # end for

    # Choose fold
    reuters_loader_train.dataset.set_fold(k)
    reuters_loader_dev.dataset.set_fold(k)
    reuters_loader_test.dataset.set_fold(k)

    # Get training data for this fold
    for i, data in enumerate(reuters_loader_train):
        # Inputs and labels
        inputs, labels, time_labels = data

        # Data length
        data_length = len(inputs)

        # Author directory
        author_train_dir = os.path.join(fold_train_dir, u"class{}".format(int(labels[0])))

        # For each segment
        cont = True
        s = 0
        file_index = 0
        while cont:
            # Segment
            segment = inputs[s:s+512]

            # Write to file
            file_desc = codecs.open(os.path.join(author_train_dir, u"file{}".format(file_index)), "w", encoding="utf-8")
            file_desc.write(u" ".join(segment).replace(u"$ ", u"$"))
            file_desc.close()

            # End ?
            if s + 512 > data_length:
                cont = False
            else:
                s += 512
            # end if

            # Inc
            file_index += 1
        # end while
    # end for

    # Get test data for this fold
    for i, data in enumerate(reuters_loader_test):
        # Inputs and labels
        inputs, labels, time_labels = data

        # Data length
        data_length = len(inputs)

        # Author directory
        author_test_dir = os.path.join(fold_test_dir, u"class{}".format(int(labels[0])))

        # For each segment
        cont = True
        s = 0
        file_index = 0
        while cont:
            # Segment
            segment = inputs[s:s + 512]

            # Write to file
            file_desc = codecs.open(os.path.join(author_test_dir, u"file{}".format(file_index)), "w", encoding="utf-8")
            file_desc.write(u" ".join(segment).replace(u"$ ", u"$"))
            file_desc.close()

            # End ?
            if s + 512 > data_length:
                cont = False
            else:
                s += 512
            # end if

            # Inc
            file_index += 1
        # end while
    # end for
# end for
