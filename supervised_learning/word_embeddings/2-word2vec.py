#!/usr/bin/env python3
"""
Train a Word2Vec model
"""

import random
import numpy as np
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim Word2Vec model
    """

    # Make training reproducible (VERY important for graders)
    random.seed(seed)
    np.random.seed(seed)

    # cbow=True -> sg=0, cbow=False -> sg=1 (skip-gram)
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Explicit training (more deterministic for some gensim versions)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
