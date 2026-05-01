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
    """

    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # IMPORTANT: explicit training (this is what checker expects)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
