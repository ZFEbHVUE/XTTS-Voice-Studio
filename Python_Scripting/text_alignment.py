"""Ordered text matching for ASR diagnostics and occurrence-specific pitch tags.

ASR differences are evidence to inspect, not proof of what a listener hears.
"""
import re
import unicodedata
from difflib import SequenceMatcher


def words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize('NFC', text).casefold(), re.UNICODE)


def ordered_matches(expected, observed):
    """Map exact matching word positions, preserving order and repeated words."""
    matcher = SequenceMatcher(None, expected, observed, autojunk=False)
    return {a + k: b + k for a, b, size in matcher.get_matching_blocks() for k in range(size)}


def ending_metrics(said, asked, tail_size=8):
    expected, observed = words(asked), words(said)
    if not expected:
        raise ValueError("Expected text must contain words")
    if tail_size < 1:
        raise ValueError("tail_size must be positive")
    matches = ordered_matches(expected, observed)
    tail_start = max(0, len(expected) - tail_size)
    kept = sum(i in matches for i in range(tail_start, len(expected)))
    missing_suffix = len(expected) - 1 - max(matches, default=-1)
    return dict(tail_kept=kept / (len(expected) - tail_start),
                missing_suffix=missing_suffix, word_ratio=len(observed) / len(expected),
                matched_fraction=len(matches) / len(expected),
                suspect=missing_suffix >= 2 or kept / (len(expected) - tail_start) < .75)
