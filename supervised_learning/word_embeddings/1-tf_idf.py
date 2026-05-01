#!/usr/bin/env python3
"""
Module that computes TF-IDF embeddings from text sentences.
"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Computes a TF-IDF embedding matrix.

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

    tf = np.zeros((s, f))

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_to_idx:
                tf[i][word_to_idx[word]] += 1

        if len(words) > 0:
            tf[i] = tf[i] / len(words)

    df = np.zeros(f)

    for word, j in word_to_idx.items():
        count = 0
        for words in tokenized:
            if word in words:
                count += 1
        df[j] = count

    idf = np.log((1 + s) / (1 + df)) + 1

    tf_idf_matrix = tf * idf

    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tf_idf_matrix = tf_idf_matrix / norms

    return tf_idf_matrix, np.array(features)
