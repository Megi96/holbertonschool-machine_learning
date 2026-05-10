#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score for a sentence.
"""

import math
from collections import Counter


def _get_ngrams(sequence, n):
    """
    Extracts n-grams from a sequence.

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


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list of list of str): reference translations
        sentence (list of str): candidate sentence
        n (int): n-gram size

    Returns:
        float: n-gram BLEU score
    """
    if len(sentence) < n:
        return 0

    # Candidate n-grams
    cand_ngrams = _get_ngrams(sentence, n)

    # Max reference n-gram counts
    max_ref_ngrams = {}

    for ref in references:
        ref_ngrams = _get_ngrams(ref, n)
        for gram, count in ref_ngrams.items():
            if gram not in max_ref_ngrams:
                max_ref_ngrams[gram] = count
            else:
                max_ref_ngrams[gram] = max(max_ref_ngrams[gram], count)

    # Clip counts
    clipped = {
        gram: min(count, max_ref_ngrams.get(gram, 0))
        for gram, count in cand_ngrams.items()
    }

    clipped_total = sum(clipped.values())
    total_ngrams = sum(cand_ngrams.values())

    precision = clipped_total / total_ngrams if total_ngrams > 0 else 0

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
        bp = math.exp(1 - (closest_ref_len / cand_len)) if cand_len > 0 else 0

    return bp * precision
