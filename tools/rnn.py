#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
from . import settings
from models.TweetRNN import TweetRNN
from models.TweetEmbRNN import TweetEmbRNN


# Create model for profiling
def create_profiling_model(feature, pretrained, cuda, embedding_dim=300, hidden_dim=1000, vocab_size=100,
                           rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64):
    """
    Create model for profiling
    :param feature:
    :param pretrained:
    :param cuda:
    :param embedding_dim:
    :param hidden_dim:
    :param vocab_size:
    :param rnn_type:
    :param num_layers:
    :param dropout:
    :param output_dropout:
    :param batch_size:
    :return:
    """
    # Feature
    if pretrained:
        rnn = TweetRNN(
            input_dim=settings.input_dims[feature],
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            output_dropout=output_dropout,
            batch_size=batch_size
        )
    else:
        # Model
        rnn = TweetEmbRNN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            output_dropout=output_dropout,
            batch_size=batch_size
        )
    # end if

    # CUDA
    if cuda:
        rnn.cuda()
    # end if

    return rnn
# end create_profiling_model


# Create model
def create_model(feature, pretrained, cuda, embedding_dim=300, hidden_dim=1000, vocab_size=100, rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64):
    # Feature
    if pretrained:
        rnn = RNN(
            input_dim=settings.input_dims[feature],
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            output_dropout=output_dropout,
            batch_size=batch_size
        )
    else:
        if "ce" in feature:
            rnn = CNNRNN(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                text_length=settings.ce_text_length,
                hidden_dim=hidden_dim,
                rnn_type=rnn_type,
                num_layers=num_layers,
                dropout=dropout,
                output_dropout=output_dropout,
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
                output_dropout=output_dropout,
                batch_size=batch_size
            )
        # end if
    # end if

    # CUDA
    if cuda:
        rnn.cuda()
    # end if

    return rnn
# end create_model
