#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.autograd import Variable


# RNN
class RNN(nn.Module):
    """
    RNN
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, n_authors=15, rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64):
        """
        Constructor
        :param input_dim:
        :param hidden_dim:
        :param n_authors:
        """
        # Super
        super(RNN, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.n_authors = n_authors
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
        self.hidden2outputs = nn.Linear(hidden_dim, n_authors)

        # Dropout
        self.dropout_layer = nn.Dropout(p=output_dropout)

        # Init hiddens
        self.hidden = self.init_hidden(batch_size)
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
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

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
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

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
    def forward(self, x, x_lengths, hidden_mask=None, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, seq_len, embedding_dim = x.size()

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size)
        # end if

        # Masked hidden
        if hidden_mask is not None:
            masked_hidden = (self.hidden[0][:, hidden_mask, :], self.hidden[1][:, hidden_mask, :])
        else:
            masked_hidden = self.hidden
        # end if

        # Pack to hide padded item to RNN
        x = utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # Exec. RNN
        x, result_hidden = self.rnn(x, masked_hidden)
        if hidden_mask is not None:
            self.hidden[0][:, hidden_mask, :] = result_hidden[0]
            self.hidden[1][:, hidden_mask, :] = result_hidden[1]
        else:
            self.hidden = result_hidden
        # end if

        # Undo packing
        x, _ = utils.rnn.pad_packed_sequence(x, batch_first=True)

        # View to (length, output size)
        x = x.contiguous()
        x = x.view((-1, self.hidden_dim))

        # Dropout
        x = self.dropout_layer(x)

        # Linear layer
        x = self.hidden2outputs(x)

        # Author scores
        x = F.log_softmax(x, dim=1)

        # Back to (batch_size, seq_len, nb_tags)
        x = x.view(batch_size, seq_len, self.n_authors)

        return x
    # end forward

# end EmbRNN
