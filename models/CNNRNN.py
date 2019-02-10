#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as utils


# RNN with CNN inputs
class CNNRNN(nn.Module):
    """
    RNN with CNN inputs
    """

    # Constructor
    def __init__(self, embedding_dim, vocab_size, text_length, hidden_dim, n_authors=15, out_channels=(50, 50, 50), kernel_sizes=(3, 4, 5), rnn_type='lstm', num_layers=1, dropout=0.0, output_dropout=0.0, batch_size=64):
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
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.text_length = text_length
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.batch_size = batch_size
        self.n_authors = n_authors
        self.output_dropout = output_dropout

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
    def forward(self, x, x_lengths, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, seq_len, input_dim = x.size()

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size)
        # end if

        # Embeddings
        x = self.embedding_layer(x)

        # Add channel dim
        x = x.contiguous()
        x = x.view(-1, self.text_length, self.embedding_dim)
        x = x.unsqueeze(1)

        # Conv window
        out_win1 = F.relu(self.conv_w1(x))
        out_win2 = F.relu(self.conv_w2(x))
        out_win3 = F.relu(self.conv_w3(x))

        # Remove last dim
        out_win1 = torch.squeeze(out_win1, dim=3)
        out_win2 = torch.squeeze(out_win2, dim=3)
        out_win3 = torch.squeeze(out_win3, dim=3)

        # Max pooling
        max_win1 = self.max_pool_w1(out_win1)
        max_win2 = self.max_pool_w2(out_win2)
        max_win3 = self.max_pool_w3(out_win3)

        # Concatenate
        x = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Back to (batch_size, seq_len, n_filters)
        x = x.view(batch_size, seq_len, -1)

        # Pack to hide padded item to RNN
        x = utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # Exec. RNN
        x, self.hidden = self.rnn(x, self.hidden)

        # Undo packing
        x, _ = utils.rnn.pad_packed_sequence(x, batch_first=True)

        # View to (length, output size)
        x = x.contiguous()
        x = x.view((-1, self.hidden_dim))

        # Author space
        x = self.hidden2outputs(x)

        # Author scores
        x = F.log_softmax(x, dim=1)

        # Back to (batch_size, seq_len, nb_tags)
        x = x.view(batch_size, seq_len, self.n_authors)

        return x
    # end forward

# end EmbRNN
