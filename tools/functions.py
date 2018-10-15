#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import echotorch.nn as etnn
import torch


#########################################
# Other
#########################################


# Manage W
def manage_w(xp, args, keep_w):
    """
    Manage W
    :param xp:
    :param args:
    :param keep_w:
    :return:
    """
    # First params
    rc_size = int(args.get_space()['reservoir_size'][0])
    rc_w_sparsity = args.get_space()['w_sparsity'][0]

    # Create W matrix
    w = etnn.ESNCell.generate_w(rc_size, rc_w_sparsity)

    # Save classifier
    if keep_w:
        xp.save_object(u"w", w)
    # end if

    return w
# end manage_w


# Get params
def get_params(space):
    """
    Get params
    :param space:
    :return:
    """
    # Params
    hidden_size = int(space['hidden_size'])
    cell_size = int(space['cell_size'])
    feature = space['feature'][0][0]
    lang = space['lang'][0][0]
    dataset_start = space['dataset_start']
    window_size = int(space['window_size'])
    learning_window = int(space['learning_window'])
    embedding_size = int(space['embedding_size'])
    rnn_type = space['rnn_type'][0][0]
    num_layers = int(space['num_layers'])
    dropout = float(space['dropout'])

    return hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout
# end get_params


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# Append sample to batch
def append_to_batch(batch_tensor, input_tensor, padding_value=0):
    """
    Append sample
    :param batch_tensor:
    :param input_tensor:
    :return:
    """
    # Dim
    if input_tensor.dim() == 2:
        # If longer, add zero to batches
        if input_tensor.size(1) > batch_tensor.size(1):
            new_batch_tensor = torch.LongTensor(batch_tensor.size(0), input_tensor.size(1)).fill_(padding_value)
            new_batch_tensor[:, :batch_tensor.size(1)] = batch_tensor
            return torch.cat((new_batch_tensor, input_tensor), dim=0)
        else:
            new_input_tensor = torch.LongTensor(1, batch_tensor.size(1)).fill_(padding_value)
            new_input_tensor[:, :input_tensor.size(1)] = input_tensor
            return torch.cat((batch_tensor, new_input_tensor), dim=0)
        # end if
    elif input_tensor.dim() == 3:
        # If longer, add zero to batch
        if input_tensor.size(1) > batch_tensor.size(1):
            new_batch_tensor = torch.FloatTensor(batch_tensor.size(0), input_tensor.size(1), input_tensor.size(2)).fill_(padding_value)
            new_batch_tensor[:, :batch_tensor.size(1)] = batch_tensor
            return torch.cat((new_batch_tensor, input_tensor), dim=0)
        else:
            new_input_tensor = torch.FloatTensor(1, batch_tensor.size(1), input_tensor.size(2)).fill_(padding_value)
            new_input_tensor[:, :input_tensor.size(1)] = input_tensor
            return torch.cat((batch_tensor, new_input_tensor), dim=0)
        # end if
    # end if
# end append_to_batch


# Transform list to tensor
def list_to_tensors(batch_list):
    """
    Transform list to tensor
    :param batch_list:
    :return:
    """
    # Batch size
    batch_size = len(batch_list)

    # Max sequence length
    max_sequence_length = batch_list[0][2]

    # Tensor type
    if batch_list[0][0].dim() == 2:
        batch_inputs = torch.LongTensor(batch_size, max_sequence_length).fill_(0)
    else:
        batch_inputs = torch.LongTensor(batch_size, max_sequence_length, batch_list[0][0].size(2)).fill_(0)
    # end if

    # Time labels
    batch_labels = torch.LongTensor(batch_size, max_sequence_length)

    # Lengths
    batch_lengths = torch.LongTensor(batch_size)

    # Timeless labels
    batch_timeless_labels = torch.LongTensor(batch_size)

    # For each sample
    for i, (inputs, time_labels, sequence_length, label) in enumerate(batch_list):
        # Set inputs
        batch_inputs[i, :inputs.size(1)] = inputs

        # Set labels
        batch_labels[i].fill_(time_labels[0][0])

        # Set length
        batch_lengths[i] = sequence_length

        # Timeless label
        batch_timeless_labels[i] = label
    # end for

    return batch_inputs, batch_labels, batch_lengths, batch_timeless_labels
# end list_to_tensors
