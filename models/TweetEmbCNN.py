#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.autograd import Variable


# Tweet CNN with embedding layer
class TweetEmbCNN(nn.Module):
    """
    Tweet CNN with embedding layer
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, dropout=0.0, batch_size=64):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param n_authors:
        """
        # Super
        super(TweetEmbCNN, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout

        # Hidden state to outputs
        self.hidden2outputs = nn.Linear(hidden_dim, 2)

        # Dropout
        self.dropout_layer = nn.Dropout(p=output_dropout)

        # Init hiddens
        # self.hidden = self.init_hidden(batch_size)
    # end __init__

    # Forward
    def forward(self, x, x_lengths, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, tweet_length, input_dim = x.size()
        print("Batch size : {}".format(batch_size))
        print("Tweet length : {}".format(tweet_length))
        print("Input dim : {}".format(input_dim))
        # Contiguous inputs
        # x = x.contiguous()

        # Resize to batch * n_tweets, tweet_length, embedding_dim
        # x = x.reshape(batch_size, tweet_length, self.input_dim)
        x_lengths = x_lengths.reshape(batch_size)
        print("X length : {}".format(x_lengths))
        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size)
        # end if

        # Pack to hide padded item to RNN
        x = utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

        # Exec. RNN
        x, result_hidden = self.rnn(x)
        self.hidden = result_hidden
        # print(x.size())
        # print(x)
        # Undo packing
        x, _ = utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Dropout
        x = self.dropout_layer(x)

        # Linear layer
        x = self.hidden2outputs(x)

        # Author scores
        x = F.log_softmax(x, dim=1)

        return x
    # end forward

# end SingleTweetRNN
