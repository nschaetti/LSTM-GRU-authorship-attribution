#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import sys
import torchlanguage.transforms
import os
import torch
import settings
from models import *
import torch.nn.utils as utils


# Create model
def create_model(feature, pretrained, cuda, embedding_dim=300, hidden_dim=1000, vocab_size=100, rnn_type='lstm', num_layers=1, dropout=False, batch_size=64):
    # Feature
    if pretrained:
        rnn = RNN(
            input_dim=settings.input_dims[feature],
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size
        )
    else:
        # Model
        rnn = EmbRNN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size
        )
    # end if

    # CUDA
    if cuda:
        rnn.cuda()
    # end if

    return rnn
# end create_model