#!/usr/bin/env python3
"""
Module that trains a Word2Vec model using gensim.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.

    Args:
        sentences (list of list of str): sentences to train on
        vector_size (int): embedding size
        min_count (int): minimum word count
        window (int): context window size
        negative (int): negative sampling size
        cbow (bool): True for CBOW, False for Skip-gram
        epochs (int): training iterations
        seed (int): random seed
        workers (int): number of threads

    Returns:
        gensim.models.word2vec.Word2Vec: trained model
    """

    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    model.build_vocab(sentences)

    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=epochs
    )

    return model
