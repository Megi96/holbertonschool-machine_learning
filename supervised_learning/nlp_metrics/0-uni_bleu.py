#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence.
"""

import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates unigram BLEU score.

    Args:
        references (list of list of str): Reference translations
        sentence (list of str): Candidate sentence

    Returns:
        float: unigram BLEU score
    """
    if len(sentence) == 0:
        return 0

    # Candidate word counts
    cand_counts = Counter(sentence)

    # Maximum reference counts for each word
    max_ref_counts = {}

    for ref in references:
        ref_counts = Counter(ref)
        for word, count in ref_counts.items():
            if word not in max_ref_counts:
                max_ref_counts[word] = count
            else:
                max_ref_counts[word] = max(max_ref_counts[word], count)

    # Clip counts
    clipped_counts = {
        word: min(count, max_ref_counts.get(word, 0))
        for word, count in cand_counts.items()
    }

    clipped_total = sum(clipped_counts.values())
    total_words = len(sentence)

    precision = clipped_total / total_words

    # Brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lengths,
        key=lambda ref_len: (abs(ref_len - total_words), ref_len)
    )

    if total_words > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - (closest_ref_len / total_words))

    return bp * precision
