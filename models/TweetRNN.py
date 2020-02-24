#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.autograd import Variable


# Tweet RNN
class TweetRNN(nn.Module):
    """
    Tweet RNN
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0,
                 batch_size=64):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param n_authors:
        """
        # Super
        super(TweetRNN, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.output_dropout = output_dropout

        # RNN
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        # end if

        # Hidden state to outputs
        self.hidden2outputs = nn.Linear(hidden_dim, 2)

        # Dropout
        self.dropout_layer = nn.Dropout(p=output_dropout)

        # Init hiddens
        # self.hidden = self.init_hidden(batch_size)
    # end __init__

    # Init hidden state
    def init_hidden(self, batch_size, n_tweets):
        """
        Init hidden state
        (num_layers, minibatch_size, hidden_dim)
        :return:
        """
        if self.rnn_type == 'lstm':
            # Hidden and cell
            hidden = torch.zeros(self.num_layers, batch_size * n_tweets, self.hidden_dim)
            cell = torch.zeros(self.num_layers, batch_size * n_tweets, self.hidden_dim)

            # To GPU
            if next(self.parameters()).is_cuda:
                hidden = hidden.cuda()
                cell = cell.cuda()
            # end if

            # To Variable
            hidden = Variable(hidden)
            cell = Variable(cell)

            return hidden, cell
        else:
            # Hidden
            hidden = torch.zeros(self.num_layers, batch_size * n_tweets, self.hidden_dim)

            # To GPU
            if next(self.parameters()).is_cuda:
                hidden = hidden.gpu()
            # end if

            # To variable
            hidden = Variable(hidden)

            return hidden
        # end if
    # end init_hidden

    # Forward
    def forward(self, x, hidden_mask=None, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, n_tweets, tweet_length, embedding_dim = x.size()
        # print("x: {}".format(x.size()))
        # Contiguous inputs
        x = x.contiguous()

        # Resize to batch * n_tweets, tweet_length, embedding_dim
        x = x.reshape(batch_size * n_tweets, tweet_length, embedding_dim)
        # print("x: {}".format(x.size()))
        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size, n_tweets)
        # end if
        # print("hidden: {}".format(self.hidden.size()))
        # Exec. RNN
        x, result_hidden = self.rnn(x)
        self.hidden = result_hidden
        # print("x: {}".format(x.size()))
        # Dropout
        x = self.dropout_layer(x)
        # print("x: {}".format(x.size()))
        # Linear layer
        x = self.hidden2outputs(x)
        # print("x: {}".format(x.size()))
        # Author scores
        x = F.log_softmax(x, dim=1)
        # print("x: {}".format(x.size()))
        # Resize back
        x = x.reshape(batch_size, n_tweets, tweet_length, 2)
        # print("x: {}".format(x.size()))
        # Average
        x = torch.mean(torch.mean(x, dim=2), dim=1)

        return x
    # end forward

# end TweetRNN
