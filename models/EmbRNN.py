#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# RNN with Embeddings
class EmbRNN(nn.Module):
    """
    RNN with embeddings
    """

    # Constructor
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_authors=15, rnn_type='lstm', num_layers=1, dropout=True):
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
        self.hidden = self.init_hidden()
    # end __init__

    # Init hidden state
    def init_hidden(self):
        """
        Init hidden state
        (num_layers, minibatch_size, hidden_dim)
        :return:
        """
        if self.rnn_type == 'lstm':
            return (torch.zeros(1, self.num_layers, self.hidden_dim),
                    torch.zeros(1, self.num_layers, self.hidden_dim))
        else:
            return torch.zeros(1, self.num_layers, self.hidden_dim)
        # end if
    # end init_hidden

    # Forward
    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        # Embedding
        embeds = self.embedding_layer(x)

        # View to (length, 1, emb_dim)
        embeds = embeds.view((x.size(0), 1, -1))

        # Exec. RNN
        rnn_out, self.hidden = self.rnn(embeds, self.hidden)

        # View to (length, output size)
        rnn_out = rnn_out.view((x.size(0), -1))

        # Author space
        outputs = self.hidden2outputs(rnn_out)

        # Author scores
        author_scores = F.log_softmax(outputs, dim=1)

        return author_scores
    # end forward

# end EmbRNN
