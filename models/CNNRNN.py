#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# RNN with CNN inputs
class CNNRNN(nn.Module):
    """
    RNN with CNN inputs
    """

    # Constructor
    def __init__(self, embedding_dim, vocab_size, text_length, hidden_dim, n_authors=15, out_channels=(50, 50, 50), kernel_sizes=(3, 4, 5), rnn_type='lstm', num_layers=1, dropout=True):
        """
        Constructor
        :param embedding_dim:
        :param vocab_size:
        :param text_length:
        :param hidden_dim:
        :param n_authors:
        :param out_channels:
        :param kernel_sizes:
        :param rnn_type:
        :param num_layers:
        :param dropout:
        """
        # Super
        super(CNNRNN, self).__init__()

        # Properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.text_length = text_length
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        # Embeddings
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Conv window 1
        self.conv_w1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                                 kernel_size=(kernel_sizes[0], embedding_dim))

        # Conv window 2
        self.conv_w2 = nn.Conv2d(in_channels=1, out_channels=out_channels[1],
                                 kernel_size=(kernel_sizes[1], embedding_dim))

        # Conv window 3
        self.conv_w3 = nn.Conv2d(in_channels=1, out_channels=out_channels[2],
                                 kernel_size=(kernel_sizes[2], embedding_dim))

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[0] + 1, stride=0)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[1] + 1, stride=0)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=text_length - kernel_sizes[2] + 1, stride=0)

        # RNN input size
        self.rnn_input_size = out_channels[0] + out_channels[1] + out_channels[2]

        # RNN
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.rnn_input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(self.rnn_input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(self.rnn_input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
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
        # Embeddings
        embeds = self.embeddings(x)

        # Add channel dim
        embeds = torch.unsqueeze(embeds, dim=1)

        # Conv window
        out_win1 = F.relu(self.conv_w1(embeds))
        out_win2 = F.relu(self.conv_w2(embeds))
        out_win3 = F.relu(self.conv_w3(embeds))

        # Remove last dim
        out_win1 = torch.squeeze(out_win1, dim=3)
        out_win2 = torch.squeeze(out_win2, dim=3)
        out_win3 = torch.squeeze(out_win3, dim=3)

        # Max pooling
        max_win1 = self.max_pool_w1(out_win1)
        max_win2 = self.max_pool_w2(out_win2)
        max_win3 = self.max_pool_w3(out_win3)

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Exec. RNN
        rnn_out, self.hidden = self.rnn(out, self.hidden)

        # View to (length, output size)
        rnn_out = rnn_out.view((x.size(0), -1))

        # Author space
        outputs = self.hidden2outputs(rnn_out)

        # Author scores
        author_scores = F.log_softmax(outputs, dim=1)

        return author_scores
    # end forward

# end EmbRNN
