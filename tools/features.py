#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import torchlanguage.transforms
import os
import torch
import settings


# Create tokenizer
def create_tokenizer(tokenizer_type, lang="en_core_web_lg"):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "nltk-twitter":
        tokenizer = nsNLP.tokenization.NLTKTweetTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang)
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang, original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer


# Create transformer
def create_transformer(feature, learning_window, pretrained=False, path="", lang="en_vectors_web_lg"):
    """
    Create the transformer
    :param feature:
    :param path:
    :param lang:
    :param n_gram:
    :return:
    """
    # ## Part-Of-Speech
    if "pos" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.PartOfSpeech(model='en'),
            torchlanguage.transforms.ToIndex(start_ix=1),
            torchlanguage.transforms.Reshape((-1))
        ])
    # ## Function words
    elif "fw" in feature:
        if pretrained:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.FunctionWord(model=lang, join=True),
                torchlanguage.transforms.GloveVector(model=lang),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.FunctionWord(model=lang),
                torchlanguage.transforms.ToIndex(start_ix=1),
                torchlanguage.transforms.Reshape((-1))
                # torchlanguage.transforms.ToMultipleLength(length_multi=learning_window),
                # torchlanguage.transforms.Reshape((-1, learning_window))
            ])
        # end if
    # ## Word Vector
    elif "wv" in feature:
        if pretrained:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, 'word2vec', "embedding.en.bin")
                ),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.ToIndex(start_ix=1),
                torchlanguage.transforms.Reshape((-1))
                # torchlanguage.transforms.ToMultipleLength(length_multi=learning_window),
                # torchlanguage.transforms.Reshape((-1, learning_window))
            ])
        # end if
    # ## Character embedding
    elif "c1" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character(),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character(),
                torchlanguage.transforms.ToIndex(start_ix=1),
                torchlanguage.transforms.Reshape((-1))
                # torchlanguage.transforms.ToMultipleLength(length_multi=learning_window),
                # torchlanguage.transforms.Reshape((-1, learning_window))
            ])
        # end if
    # ## Character 2-gram embedding
    elif "c2" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character2Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character2Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(start_ix=1),
                torchlanguage.transforms.Reshape((-1))
                # torchlanguage.transforms.ToMultipleLength(length_multi=learning_window),
                # torchlanguage.transforms.Reshape((-1, learning_window))
            ])
        # end if
    # ## Character 3-gram embedding
    elif "c3" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character3Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character3Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(start_ix=1),
                torchlanguage.transforms.Reshape((-1))
                # torchlanguage.transforms.ToMultipleLength(length_multi=learning_window),
                # torchlanguage.transforms.Reshape((-1, learning_window))
            ])
        # end if
    else:
        raise NotImplementedError(u"Feature type {} not implemented".format(feature))
    # end if
# end create_transformer


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding
