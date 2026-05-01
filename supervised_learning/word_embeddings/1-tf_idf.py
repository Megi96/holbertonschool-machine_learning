#!/usr/bin/env python3
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    TF-IDF = Term Frequency × Inverse Document Frequency

    Args:
        sentences (list of str): Sentences to analyze.
        vocab (list of str, optional): Vocabulary to use.
            If None, vocabulary is built from sentences.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): TF-IDF matrix of shape (s, f)
            - features (numpy.ndarray): Vocabulary used
    """

    tokenized = []

    # --- Step 1: preprocess sentences ---
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r"\'s\b", "", sentence)
        words = re.findall(r"\b\w+\b", sentence)
        tokenized.append(words)

    # --- Step 2: build vocabulary ---
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

    # --- Step 3: compute term frequency (TF) ---
    tf = np.zeros((s, f))

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_to_idx:
                tf[i][word_to_idx[word]] += 1

        # normalize TF by sentence length
        if len(words) > 0:
            tf[i] = tf[i] / len(words)

    # --- Step 4: compute document frequency (DF) ---
    df = np.zeros(f)

    for word, j in word_to_idx.items():
        count = 0
        for words in tokenized:
            if word in words:
                count += 1
        df[j] = count

    # --- Step 5: compute IDF ---
    idf = np.log((1 + s) / (1 + df)) + 1

    # --- Step 6: TF-IDF ---
    tf_idf_matrix = tf * idf

    # --- Step 7: L2 normalization ---
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    tf_idf_matrix = tf_idf_matrix / norms

    return tf_idf_matrix, np.array(features)
