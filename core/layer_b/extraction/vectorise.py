"""Vectoriser setup, document-frequency helpers, and pattern quality filters.

This module provides:
  1. make_vectorizers()  — builds word + char CountVectorizers
  2. doc_freq()          — sparse matrix → per-feature doc counts
  3. pattern_shape_stats() + quality filters — decide if a candidate pattern
     is real signal or just common English / char-boundary noise
"""
import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

log = logging.getLogger(__name__)

STOPWORDS = set(ENGLISH_STOP_WORDS)

# Characters we strip from token edges when cleaning for stopword checks
TOKEN_STRIP_CHARS = "\"'`.,:;!?-()[]{}<>"
EDGE_PUNCT_CHARS = set(TOKEN_STRIP_CHARS)


#Vectorizer factory 

def make_vectorizers(min_df, max_features):
    """Return (word_vectorizer, char_vectorizer).

    Both expect already-normalised text (lowercase, punctuation stripped,
    whitespace collapsed) so we disable sklearn's own preprocessing.
    """
    word_vec = CountVectorizer(
        analyzer="word",
        ngram_range=(2, 5),       # 2-5 word phrases — captures attack phrases
        binary=True,              # we only care about presence, not count
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None,
        dtype=np.int8,            # saves ~8x RAM vs float64
    )
    char_vec = CountVectorizer(
        analyzer="char",
        ngram_range=(10, 12),     # 10-12 char fragments — shorter ones are too noisy
        binary=True,
        min_df=min_df,
        max_features=max_features,
        lowercase=False,
        dtype=np.int8,
    )
    return word_vec, char_vec


# ── Document frequency ───────────────────────────────────────────────────

def doc_freq(X):
    """Count how many documents each feature appears in. Returns int32 array."""
    return np.asarray(X.getnnz(axis=0)).ravel().astype(np.int32)


# ── Pattern shape analysis ───────────────────────────────────────────────
# These stats are content-agnostic — they look at structural properties of
# a pattern (how much punctuation, how many stopwords, etc.) rather than
# the actual words. This keeps the filters statistical, not keyword-based.

def _clean_token(token):
    """Strip punctuation from edges and lowercase a token."""
    return token.strip(TOKEN_STRIP_CHARS).lower()


def pattern_shape_stats(pattern):
    """Compute simple structural stats for a candidate pattern.

    Returns a dict with keys like 'alpha_ratio', 'stopword_ratio', etc.
    Used by the quality filters below to weed out noisy patterns.
    """
    stripped = pattern.strip()
    raw_tokens = stripped.split()
    clean_tokens = [_clean_token(t) for t in raw_tokens]
    clean_tokens = [t for t in clean_tokens if t]  # drop empties

    total_chars = len(stripped)
    alpha_chars = sum(c.isalpha() for c in stripped)
    punct_chars = sum(not c.isalnum() and not c.isspace() for c in stripped)
    digit_chars = sum(c.isdigit() for c in stripped)
    stopword_count = sum(t in STOPWORDS for t in clean_tokens)
    capitalized_count = sum(t[:1].isupper() for t in raw_tokens)

    # Does the pattern start or end with punctuation? (char n-gram boundary noise)
    edge_punct = 0
    if stripped and stripped[0] in EDGE_PUNCT_CHARS:
        edge_punct += 1
    if len(stripped) > 1 and stripped[-1] in EDGE_PUNCT_CHARS:
        edge_punct += 1

    n_clean = len(clean_tokens)
    n_raw = len(raw_tokens)

    return {
        "token_count":              float(n_raw),
        "alpha_ratio":              alpha_chars / total_chars if total_chars else 0.0,
        "punct_ratio":              punct_chars / total_chars if total_chars else 0.0,
        "digit_ratio":              digit_chars / total_chars if total_chars else 0.0,
        "stopword_ratio":           stopword_count / n_clean if n_clean else 1.0,
        "capitalized_token_ratio":  capitalized_count / n_raw if n_raw else 0.0,
        "edge_punct_count":         float(edge_punct),
        "unique_token_ratio":       len(set(clean_tokens)) / n_clean if n_clean else 0.0,
        "mean_token_len":           float(np.mean([len(t) for t in clean_tokens])) if n_clean else 0.0,
    }

