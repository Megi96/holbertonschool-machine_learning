#!/usr/bin/env python3
"""Question answering loop using a reference document."""

import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf


tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad'
)

model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")


def question_answer(question, reference):
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

    outputs = model([
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


def answer_loop(reference):
    """
    Starts an interactive QA loop using a reference document.

    Args:
        reference (str): Reference text containing answers.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        answer = question_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:', answer)
