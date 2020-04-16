#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage
import torchlanguage.datasets
import torchlanguage.transforms
import torch
from . import settings


# Load precomputed SFGram dataset
def load_sfgram_precomputed_dataset(author, batch_size, feature, pretrained, block_length=40, k=5, dev_per_file=False):
    """
    Load precomputed SFGram dataset
    :param author:
    :param batch_size:
    :param feature:
    :param pretrained:
    :param block_length:
    :param k:
    :return:
    """
    # Training
    sfgram_loader_train = torch.utils.data.DataLoader(
        torchlanguage.datasets.SFGramBlockDataset(
            root='sfgram_blocks',
            author=author,
            block_length=block_length,
            trained=not pretrained,
            feature=feature,
            k=k,
            set='train'
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # Validation
    sfgram_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.datasets.SFGramBlockDataset(
            root='sfgram_blocks',
            author=author,
            block_length=block_length,
            trained=not pretrained,
            feature=feature,
            k=k,
            set='dev',
            per_file=dev_per_file
        ),
        batch_size=1 if dev_per_file else batch_size,
        shuffle=True
    )

    # Test
    sfgram_loader_test = torch.utils.data.DataLoader(
        torchlanguage.datasets.SFGramBlockDataset(
            root='sfgram_blocks',
            author=author,
            block_length=block_length,
            trained=not pretrained,
            feature=feature,
            k=k,
            set='test',
            per_file=True
        ),
        batch_size=1,
        shuffle=False
    )

    return sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test
# end load_sfgram_precomputed_dataset


# Load SFGram dataset
def load_sfgram_dataset(author, batch_size, load_type, block_length=40, k=5):
    """
    Load SFGram dataset
    :param k:
    :return:
    """
    # Load
    sfgram_dataset = torchlanguage.datasets.SFGramDataset(
        download=True,
        root='data',
        author=author,
        block_length=block_length,
        stream=True,
        load_type=load_type
    )

    # Training
    sfgram_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(sfgram_dataset, train='train', k=k),
        batch_size=batch_size,
        shuffle=False
    )

    # Validation
    sfgram_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(sfgram_dataset, train='dev', k=k),
        batch_size=batch_size,
        shuffle=False
    )

    # Test
    sfgram_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(sfgram_dataset, train='test', k=k),
        batch_size=batch_size,
        shuffle=False
    )

    return sfgram_dataset, sfgram_loader_train, sfgram_loader_dev, sfgram_loader_test
# end load_sfgram_dataset


# Load PAN17 dataset
def load_pan17_dataset_per_tweet(output_length, output_dim, batch_size, trained, load_type, transform, k=10):
    """
        Load PAN 17 dataset
        :param k:
        :return:
        """
    # Load
    pan17_dataset_per_tweet = torchlanguage.datasets.PAN17AuthorProfiling(
        lang='en',
        download=True,
        root='pan17',
        output_type='long',
        outputs_length=output_length,
        output_dim=output_dim,
        trained=trained,
        shuffle=True,
        per_tweet=True,
        load_type=load_type,
        transform=transform,
        save_transform=True
    )

    # Load
    pan17_dataset = torchlanguage.datasets.PAN17AuthorProfiling(
        lang='en',
        download=True,
        root='pan17',
        output_type='long',
        outputs_length=output_length,
        output_dim=output_dim,
        trained=trained,
        shuffle=True,
        per_tweet=False,
        load_type=load_type,
        transform=transform
    )

    # Training
    pan17_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset_per_tweet, train='train', k=k),
        batch_size=batch_size,
        shuffle=True
    )

    # Validation
    pan17_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='dev', k=k),
        batch_size=batch_size,
        shuffle=False
    )

    # Test
    pan17_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='test', k=k),
        batch_size=batch_size,
        shuffle=False
    )

    return pan17_dataset, pan17_dataset_per_tweet, pan17_loader_train, pan17_loader_dev, pan17_loader_test
# end load_pan17_dataset


# Load PAN17 dataset
def load_pan17_dataset(output_length, output_dim, batch_size, trained, load_type, per_tweet=False, k=10):
    """
    Load PAN 17 dataset
    :param k:
    :return:
    """
    # Load
    pan17_dataset = torchlanguage.datasets.PAN17AuthorProfiling(
        lang='en',
        download=True,
        root='pan17',
        output_type='long',
        outputs_length=output_length,
        output_dim=output_dim,
        trained=trained,
        shuffle=True,
        per_tweet=per_tweet,
        load_type=load_type,
        save_transform=True
    )

    # Training
    pan17_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='train', k=k),
        batch_size=batch_size,
        shuffle=True if per_tweet else False
    )

    # Validation
    pan17_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='dev', k=k),
        batch_size=batch_size,
        shuffle=True if per_tweet else False
    )

    # Test
    pan17_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='test', k=k),
        batch_size=batch_size,
        shuffle=True if per_tweet else False
    )

    return pan17_dataset, pan17_loader_train, pan17_loader_dev, pan17_loader_test
# end load_pan17_dataset


# Load dataset
def load_dataset(dataset_size=100, dataset_start=0, shuffle=True, sentence_level=False, n_authors=15, k=5, features=u""):
    """
    Load dataset
    :return:
    """
    # Load from directory
    if sentence_level:
        reutersc50_dataset = torchlanguage.datasets.ReutersC50SentenceDataset(
            n_authors=n_authors,
            download=True,
            dataset_size=dataset_size,
            dataset_start=dataset_start
        )
    else:
        reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
            n_authors=n_authors,
            download=True,
            dataset_size=dataset_size,
            dataset_start=dataset_start,
            load_features=features
        )
    # end if

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='train', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    # Reuters C50 dataset dev
    reuters_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='dev', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='test', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    return reutersc50_dataset, reuters_loader_train, reuters_loader_dev, reuters_loader_test
# end load_dataset


# Load pretrain dataset
def load_pretrain_dataset():
    """
    Load dataset
    :return:
    """
    # Load from directory
    reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
        n_authors=35,
        download=True,
        dataset_size=100,
        dataset_start=0,
        authors=settings.pretrain_authors
    )

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset),
        batch_size=1,
        shuffle=True
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset, train=False),
        batch_size=1,
        shuffle=True
    )
    return reutersc50_dataset, reuters_loader_train, reuters_loader_test
# end load_dataset

