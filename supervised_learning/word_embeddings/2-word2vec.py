#!/usr/bin/env python3
"""
Train a Word2Vec model
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim Word2Vec model

    sentences: list of tokenized sentences
    vector_size: dimensionality of embeddings
    min_count: minimum word frequency
    window: context window size
    negative: negative sampling size
    cbow: True for CBOW, False for Skip-gram
    epochs: training iterations
    seed: random seed
    workers: number of threads

    Returns: trained Word2Vec model
    """

    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
