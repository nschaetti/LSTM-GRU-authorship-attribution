#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


# Imports
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.autograd import Variable


# RNN with Tweet Embeddings
class TweetEmbRNN(nn.Module):
    """
    Tweet RNN with embeddings
    """

    # Constructor
    def __init__(self, embedding_dim, hidden_dim, vocab_size, rnn_type='lstm', num_layers=1, dropout=0.0,
                 output_dropout=0.0, batch_size=64):
        """
        Constructor
        :param embedding_dim:
        :param hidden_dim:
        :param vocab_size:
        :param n_authors:
        """
        # Super
        super(TweetEmbRNN, self).__init__()

        # Properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # Embeddings
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

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

        # Hidden state to outputs
        self.hidden2outputs = nn.Linear(hidden_dim, 2)

        # Dropout
        self.dropout_layer = nn.Dropout(p=output_dropout)
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
            hidden = torch.randn(self.num_layers, batch_size * n_tweets, self.hidden_dim)
            cell = torch.randn(self.num_layers, batch_size * n_tweets, self.hidden_dim)

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
            hidden = torch.randn(self.num_layers, batch_size * n_tweets, self.hidden_dim)

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
    def forward(self, x, x_lengths, reset_hidden=True):
        """
        Forward pass
        :param x:
        :return:
        """
        # Sizes
        batch_size, n_tweets, tweet_length = x.size()

        # Max. length
        max_tweet_length = torch.max(x_lengths).item()

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size, n_tweets)
        # end if

        # Embedding
        x = self.embedding_layer(x)

        # Contiguous inputs
        x = x.contiguous()

        # Resize to batch * n_tweets, tweet_length, embedding_dim
        x = x.reshape(batch_size * n_tweets, tweet_length, self.embedding_dim)
        x_lengths = x_lengths.reshape(batch_size * n_tweets)

        # Init hiddens
        if reset_hidden:
            self.hidden = self.init_hidden(batch_size, n_tweets)
        # end if

        # Pack to hide padded item to RNN
        x = utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)

        # Exec. RNN
        x, result_hidden = self.rnn(x)
        self.hidden = result_hidden

        # Undo packing
        x, _ = utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Contiguous
        x = x.contiguous()

        # Dropout
        x = self.dropout_layer(x)

        # Linear layer
        x = self.hidden2outputs(x)

        # Author scores
        x = F.log_softmax(x, dim=1)

        # Resize back
        x = x.reshape(batch_size, n_tweets, max_tweet_length, 2)

        # Average
        x = torch.mean(torch.mean(x, dim=2), dim=1)

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
