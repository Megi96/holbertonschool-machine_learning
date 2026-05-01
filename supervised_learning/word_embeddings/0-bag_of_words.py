#!/usr/bin/env python3
"""
Module that creates a bag of words embedding matrix.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences (list of str): sentences to analyze
        vocab (list of str): vocabulary words to use

    Returns:
        tuple: embeddings, features
            embeddings is a numpy.ndarray of shape (s, f)
            features is a numpy.ndarray of vocabulary words
    """

    tokenized = []

    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"\'s\b", "", sentence)
        words = re.findall(r"\b\w+\b", sentence)
        tokenized.append(words)

    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        features = sorted(list(vocab_set))
    else:
        features = vocab

    word_to_idx = {word: i for i, word in enumerate(features)}

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_to_idx:
                embeddings[i][word_to_idx[word]] += 1

    return embeddings, np.array(features)
