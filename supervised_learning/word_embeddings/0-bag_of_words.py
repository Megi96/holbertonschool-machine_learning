#!/usr/bin/env python3
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    The function tokenizes a list of sentences, builds a vocabulary
    (or uses a provided one), and returns a matrix where each row
    represents a sentence and each column represents a word count.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): List of vocabulary words to use.
            If None, vocabulary is built from the input sentences.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): Matrix of shape (s, f) where
              s is number of sentences and f is number of features.
            - features (numpy.ndarray): Array of vocabulary words used.
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
