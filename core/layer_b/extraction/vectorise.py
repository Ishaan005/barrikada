"""Vectoriser factory and document-frequency helpers for signature extraction."""
import logging

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

log = logging.getLogger(__name__)

STOPWORDS = set(ENGLISH_STOP_WORDS)


def make_vectorizers(min_df: int, max_features: int):
    """Return (word_vectorizer, char_vectorizer) with scale-aware parameters.

    Both vectorisers operate on already-normalised text (lowercase,
    punctuation stripped, whitespace collapsed) so ``lowercase=False`` and
    no additional preprocessor is needed.
    """
    word_vec = CountVectorizer(
        analyzer="word",
        ngram_range=(1, 4),
        binary=True,          # presence/absence only
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None,
        dtype=np.int8,        # ~8× less RAM than float64
    )
    char_vec = CountVectorizer(
        analyzer="char",
        ngram_range=(6, 8),
        binary=True,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        dtype=np.int8,
    )
    return word_vec, char_vec


def doc_freq(X: csr_matrix) -> np.ndarray:
    """Return per-feature document frequency as int32."""
    return np.asarray(X.getnnz(axis=0)).ravel().astype(np.int32)


def is_stopwordy_ngram(pattern: str) -> bool:
    """Return True if pattern is a single token or entirely stop-words."""
    tokens = pattern.split()
    if len(tokens) < 2:
        return True
    return all(t in STOPWORDS for t in tokens)


# Common first-person / pronoun starts that appear in both safe and malicious text.
_GENERIC_WORD_PATTERNS = {
    "i want", "i have", "i need", "i would", "i will", "i am", "i can",
    "i think", "i know", "i believe", "i like", "i could", "i should",
    "i was", "i did", "i do", "i had", "i got", "i see", "i feel",
    "you are", "you can", "you have", "you need", "you should", "you will",
    "you would", "you want", "you may", "you must", "you might",
    "it is", "it was", "it has", "it can", "it will", "it would",
    "this is", "that is", "there is", "there are", "here is", "here are",
    "we have", "we are", "we can", "we need", "we will", "we should",
    "they are", "they have", "they can", "they will", "they were",
    "he is", "he was", "he has", "she is", "she was", "she has",
}


def is_generic_pattern(pattern: str) -> bool:
    """Return True if the pattern is too generic to be a useful malicious signature.

    Catches:
      - Stopword-only n-grams
      - Common pronoun/verb starts (e.g., "i want", "you are")
      - Short char fragments that are just punctuation + common word stubs
    """
    stripped = pattern.strip()
    if not stripped:
        return True

    tokens = stripped.split()

    # Single-token patterns
    if len(tokens) < 2:
        # For word-level: always generic
        # For char-level: check if it's mostly punctuation/spaces
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        return alpha_chars <= 3

    # Pure stopword n-grams
    if all(t in STOPWORDS for t in tokens):
        return True

    # Known generic word patterns (exact match or starts-with for longer n-grams)
    lower = stripped.lower()
    for gp in _GENERIC_WORD_PATTERNS:
        if lower == gp or lower.startswith(gp + " "):
            return True

    return False
