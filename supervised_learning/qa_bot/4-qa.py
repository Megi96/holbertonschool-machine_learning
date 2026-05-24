#!/usr/bin/env python3
"""Multi-reference question answering module."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)

qa_model = hub.load(
    "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
)

embed_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the corpus directory.
        sentence (str): Sentence to compare against documents.

    Returns:
        str: Most semantically similar document.
    """
    documents = []

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            path = os.path.join(corpus_path, filename)

            with open(path, 'r', encoding='utf-8') as file:
                documents.append(file.read())

    doc_embeddings = embed_model(documents)
    sentence_embedding = embed_model([sentence])

    similarities = tf.keras.losses.cosine_similarity(
        sentence_embedding,
        doc_embeddings
    )

    best_index = np.argmin(similarities)

    return documents[best_index]


def qa_model_answer(question, reference):
    """
    Finds an answer to a question within a reference document.

    Args:
        question (str): Question to answer.
        reference (str): Reference document.

    Returns:
        str: Extracted answer.
        None: If no valid answer is found.
    """
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    tokens = (
        ['[CLS]'] +
        question_tokens +
        ['[SEP]'] +
        reference_tokens +
        ['[SEP]']
    )

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_word_ids = tf.constant([input_ids])
    input_mask = tf.constant([[1] * len(input_ids)])

    type_ids = (
        [0] * (len(question_tokens) + 2) +
        [1] * (len(reference_tokens) + 1)
    )

    input_type_ids = tf.constant([type_ids])

    outputs = qa_model([
        input_word_ids,
        input_mask,
        input_type_ids
    ])

    start_scores, end_scores = outputs

    start_index = tf.argmax(start_scores[0]).numpy()
    end_index = tf.argmax(end_scores[0]).numpy()

    if start_index >= end_index:
        return None

    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer.strip() == '':
        return None

    return answer


def question_answer(corpus_path):
    """
    Answers questions using multiple reference documents.

    Args:
        corpus_path (str): Path to the corpus directory.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)

        answer = qa_model_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:', answer)
