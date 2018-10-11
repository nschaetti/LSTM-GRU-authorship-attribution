#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import echotorch.nn as etnn


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
    print(space)
    # Params
    hidden_size = int(space['hidden_size'])
    cell_size = int(space['cell_size'])
    feature = space['feature'][0][0]
    lang = space['lang'][0][0]
    dataset_start = space['dataset_start']
    window_size = int(space['window_size'])
    learning_window = int(space['learning_window'])

    return hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window
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
