#!/usr/bin/env python3
"""Semantic search module."""

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the corpus directory.
        sentence (str): Sentence to compare against documents.

    Returns:
        str: Reference text of the most similar document.
    """
    documents = []
    filenames = []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)

            with open(path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)

    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )

    doc_embeddings = embed(documents)
    sentence_embedding = embed([sentence])

    similarities = tf.keras.losses.cosine_similarity(
        sentence_embedding,
        doc_embeddings
    )

    similarities = np.array(similarities)

    best_index = np.argmin(similarities)

    return documents[best_index]
