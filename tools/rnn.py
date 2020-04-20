#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
from . import settings
from models.TweetRNN import TweetRNN
from models.TweetEmbRNN import TweetEmbRNN
from models.VerificationRNN import VerificationRNN
from models.SingleTweetEmbRNN import SingleTweetEmbRNN
from models.SingleTweetRNN import SingleTweetRNN
import echotorch.nn as etnn
import echotorch.utils.matrix_generation


# Create model for author verification
def create_verification_model(feature, pretrained, cuda, embedding_dim=300, hidden_dim=100, vocab_size=10000,
                              rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64):
    """
    Create model for verification
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
        rnn = VerificationRNN(
            embedding_dim=settings.input_dims[feature],
            use_embedding_layer=False,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            output_dropout=output_dropout,
            batch_size=batch_size
        )
    else:
        # Model
        rnn = VerificationRNN(
            embedding_dim=embedding_dim,
            use_embedding_layer=True,
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
# end create_verification_model

# Create ESN model for profiling
def create_esn_profiling_model(feature, hidden_dim, connectivity, spectral_radius, leaky_rate, input_scaling=0.25,
                               bias_scaling=0.25, ridge_param=0.0001):
    """
    Create ESN model for
    :param feature:
    :param input_dim:
    :param hidden_dim:
    :param vocab_size:
    :param batch_size:
    :return:
    """
    # Internal matrix
    w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
        connectivity=connectivity,
        spetral_radius=spectral_radius
    )

    # Input weights
    win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
        connectivity=connectivity,
        scale=input_scaling,
        apply_spectral_radius=False
    )

    # Bias vector
    wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
        connectivity=connectivity,
        scale=bias_scaling,
        apply_spectral_radius=False
    )

    # ESN cell
    esn = etnn.LiESN(
        input_dim=settings.input_dims[feature],
        output_dim=2,
        hidden_dim=hidden_dim,
        leaky_rate=leaky_rate,
        ridge_param=ridge_param,
        w_generator=w_generator,
        win_generator=win_generator,
        wbias_generator=wbias_generator
    )
    return esn
# end create_esn_profiling_model

# Create model for profiling
def create_profiling_model(feature, pretrained, cuda, embedding_dim=300, hidden_dim=1000, vocab_size=100,
                           rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64,
                           per_tweet=False):
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
    # Per tweet
    if not per_tweet:
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
    else:
        # Feature
        if pretrained:
            rnn = SingleTweetRNN(
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
            rnn = SingleTweetEmbRNN(
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
