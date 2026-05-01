#!/usr/bin/env python3
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences
        vocab: list of vocabulary words (optional)

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: list of vocabulary words
    """

    # --- Step 1: preprocess sentences ---
    tokenized = []
    for sentence in sentences:
        # lowercase + remove non-alphanumeric characters
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokenized.append(words)

    # --- Step 2: build vocabulary ---
    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        features = sorted(vocab_set)
    else:
        features = vocab

    # map word → index
    word_to_idx = {word: i for i, word in enumerate(features)}

    # --- Step 3: build embedding matrix ---
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, np.array(features)
