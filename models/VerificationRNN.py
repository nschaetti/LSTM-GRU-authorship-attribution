#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# RNN for Author Verification
class VerificationRNN(nn.Module):
    """
    RNN for Author Verification
    """

    # Constructor
    def __init__(self, embedding_dim, hidden_dim, vocab_size, rnn_type='lstm', num_layers=1, dropout=0.0,
                 output_dropout=0.0, use_embedding_layer=False, batch_size=64):
        """
        Constructor
        :param embedding_dim:
        :param hidden_dim:
        :param vocab_size:
        :param n_authors:
        """
        # Super
        super(VerificationRNN, self).__init__()

        # Properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.use_embedding_layer = use_embedding_layer

        # Embeddings
        if use_embedding_layer:
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # end if

        # RNN
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True
            )
        # end if

        # Dropout
        self.dropout_layer = nn.Dropout(p=output_dropout)

        # Hidden state to outputs
        self.hidden2outputs = nn.Linear(hidden_dim, 1)
    # end __init__

    # Init hidden state
    def init_hidden(self, batch_size):
        """
        Init hidden state
        (num_layers, minibatch_size, hidden_dim)
        :return:
        """
        if self.rnn_type == 'lstm':
            # Hidden and cell
            hidden = torch.randn(self.num_layers, batch_size, self.hidden_dim)
            cell = torch.randn(self.num_layers, batch_size, self.hidden_dim)

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
            hidden = torch.randn(self.num_layers, batch_size, self.hidden_dim)

            # To GPU
            if next(self.parameters()).is_cuda:
                hidden = hidden.cuda()
            # end if

            # To variable
            hidden = Variable(hidden)

            return hidden
        # end if
    # end init_hidden

    # Forward
    def forward(self, x, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        if not self.use_embedding_layer:
            batch_size, segment_length, input_dim = x.size()
        else:
            batch_size, segment_length = x.size()
        # end if

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size)
        # end if

        # Embedding
        if self.use_embedding_layer:
            x = self.embedding_layer(x)
        # end if

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size)
        # end if

        # Exec. RNN
        x, result_hidden = self.rnn(x)
        self.hidden = result_hidden

        # Dropout
        x = self.dropout_layer(x)

        # Linear layer
        x = self.hidden2outputs(x)

        # Author scores
        x = torch.sigmoid(x)

        return x
    # end forward

# end VerificationRNN
