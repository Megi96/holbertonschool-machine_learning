#!/usr/bin/env python3
"""
Module that trains a Word2Vec model using gensim.
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.

    Args:
        sentences (list of list of str): sentences to train on
        vector_size (int): dimensionality of embeddings
        min_count (int): minimum word count
        window (int): context window size
        negative (int): negative sampling size
        cbow (bool): True for CBOW, False for Skip-gram
        epochs (int): number of training epochs
        seed (int): random seed
        workers (int): number of worker threads

    Returns:
        Word2Vec: trained model
    """

    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    model.train(sentences, total_examples=len(sentences),
                epochs=epochs)

    return model
