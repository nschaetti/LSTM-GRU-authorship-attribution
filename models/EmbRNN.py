#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable


# RNN with Embeddings
class EmbRNN(nn.Module):
    """
    RNN with embeddings
    """

    # Constructor
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_authors=15, rnn_type='lstm', num_layers=1, dropout=True, batch_size=64):
        """
        Constructor
        :param embedding_dim:
        :param hidden_dim:
        :param vocab_size:
        :param n_authors:
        """
        # Super
        super(EmbRNN, self).__init__()

        # Properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.n_authors = n_authors

        # Embeddings
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # RNN
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        # end if

        # Hidden state to outputs
        self.hidden2outputs = nn.Linear(hidden_dim, n_authors)

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
    def forward(self, x, x_lengths):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, seq_len = x.size()

        # Init hiddens
        self.hidden = self.init_hidden(batch_size)

        # Embedding
        x = self.embedding_layer(x)

        # Pack to hide padded item to RNN
        x = utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # Exec. RNN
        x, self.hidden = self.rnn(x, self.hidden)

        # Undo packing
        x, _ = utils.rnn.pad_packed_sequence(x, batch_first=True)

        # View to (length, output size)
        x = x.contiguous()
        x = x.view((-1, self.hidden_dim))

        # Linear layer
        x = self.hidden2outputs(x)

        # Author scores
        x = F.log_softmax(x, dim=1)

        # Back to (batch_size, seq_len, nb_tags)
        x = x.view(batch_size, seq_len, self.n_authors)

        return x
    # end forward

    # Loss
    def loss(self, out, labels, seq_lengths):
        """
        Loss
        :param out:
        :param labels:
        :param seq_lengths:
        :return:
        """
        # Flatten labels
        labels = labels.view(-1)

        # Flatten predictions
        out = out.view(-1, self.n_authors)

        # Create mask
        tag_pad_token = -1
        mask = (labels > tag_pad_token).float()
    # end loss

# end EmbRNN
