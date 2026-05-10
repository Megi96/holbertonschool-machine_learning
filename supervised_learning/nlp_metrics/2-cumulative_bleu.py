#!/usr/bin/env python3
"""
Calculates the cumulative n-gram BLEU score for a sentence.
"""

import math
from collections import Counter


def _get_ngrams(sequence, n):
    """
    Extract n-grams from a sequence.

    Args:
        sequence (list of str): input tokens
        n (int): n-gram size

    Returns:
        Counter: n-gram counts
    """
    return Counter(
        tuple(sequence[i:i + n])
        for i in range(len(sequence) - n + 1)
    )


def _modified_precision(references, sentence, n):
    """
    Computes modified n-gram precision with clipping.
    """
    cand_ngrams = _get_ngrams(sentence, n)

    if sum(cand_ngrams.values()) == 0:
        return 0

    max_ref_ngrams = {}

    for ref in references:
        ref_ngrams = _get_ngrams(ref, n)
        for gram, count in ref_ngrams.items():
            max_ref_ngrams[gram] = max(
                max_ref_ngrams.get(gram, 0),
                count
            )

    clipped = {
        gram: min(count, max_ref_ngrams.get(gram, 0))
        for gram, count in cand_ngrams.items()
    }

    return sum(clipped.values()) / sum(cand_ngrams.values())


def cumulative_bleu(references, sentence, n):
    """
    Calculates cumulative n-gram BLEU score.

    Args:
        references (list of list of str): reference translations
        sentence (list of str): candidate sentence
        n (int): maximum n-gram size

    Returns:
        float: cumulative BLEU score
    """
    if len(sentence) == 0:
        return 0

    precisions = []

    for i in range(1, n + 1):
        p = _modified_precision(references, sentence, i)
        if p == 0:
            return 0
        precisions.append(p)

    # Geometric mean (equal weights)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)

    # Brevity penalty
    ref_lens = [len(ref) for ref in references]
    cand_len = len(sentence)

    closest_ref_len = min(
        ref_lens,
        key=lambda ref_len: (abs(ref_len - cand_len), ref_len)
    )

    if cand_len > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - (closest_ref_len / cand_len))

    return bp * geo_mean
