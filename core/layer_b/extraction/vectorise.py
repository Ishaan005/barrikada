"""Vectoriser setup, document-frequency helpers, and pattern quality filters.

This module provides:
  1. make_vectorizers()  — builds word + char CountVectorizers
  2. doc_freq()          — sparse matrix → per-feature doc counts
  3. pattern_shape_stats() + quality filters — decide if a candidate pattern
     is real signal or just common English / char-boundary noise
"""
import logging

import numpy as np
from scipy.sparse import csr_matrix
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


# ── Quality threshold computation ────────────────────────────────────────

def _quantile(values, q, default):
    """Compute the q-th quantile of values, returning default if empty."""
    vals = list(values)
    if not vals:
        return default
    return float(np.quantile(np.asarray(vals, dtype=float), q))


def compute_statistical_quality_thresholds(patterns, supports, role):
    """Derive pattern-shape cutoffs from the candidate pool.

    Looks at the distribution of shape stats across all candidates, then
    picks percentile-based floors/caps. This way the thresholds adapt to
    whatever patterns the vectorizer found, rather than being hard-coded.

    'role' is either "safe" or "malicious" — safe patterns are more lenient
    on stopwords but stricter on capitalisation.
    """
    all_stats = [pattern_shape_stats(p) for p in patterns]

    # If there are no candidates, return permissive defaults
    if not all_stats:
        return {
            "alpha_ratio_floor": 0.0,
            "punct_ratio_cap": 1.0,
            "stopword_ratio_cap": 1.0,
            "capitalized_ratio_cap": 1.0,
            "edge_punct_cap": 2.0,
            "unique_token_ratio_floor": 0.0,
            "digit_ratio_cap": 1.0,
            "mean_token_len_floor": 0.0,
            "support_floor": 0.0,
        }

    # Pull out each metric into a flat list for percentile computation
    alpha_ratios        = [s["alpha_ratio"] for s in all_stats]
    punct_ratios        = [s["punct_ratio"] for s in all_stats]
    stopword_ratios     = [s["stopword_ratio"] for s in all_stats]
    capitalized_ratios  = [s["capitalized_token_ratio"] for s in all_stats]
    edge_punct_counts   = [s["edge_punct_count"] for s in all_stats]
    unique_token_ratios = [s["unique_token_ratio"] for s in all_stats]
    digit_ratios        = [s["digit_ratio"] for s in all_stats]
    mean_token_lens     = [s["mean_token_len"] for s in all_stats]

    # Safe patterns: allow more stopwords (0.90 quantile) but fewer caps (0.85)
    # Malicious patterns: stricter on stopwords (0.85) but more caps ok (0.95)
    if role == "safe":
        stopword_q = 0.90
        capitalized_q = 0.85
    else:
        stopword_q = 0.85
        capitalized_q = 0.95

    return {
        "alpha_ratio_floor":        _quantile(alpha_ratios, 0.10, 0.0),
        "punct_ratio_cap":          _quantile(punct_ratios, 0.90, 1.0),
        "stopword_ratio_cap":       _quantile(stopword_ratios, stopword_q, 1.0),
        "capitalized_ratio_cap":    _quantile(capitalized_ratios, capitalized_q, 1.0),
        "edge_punct_cap":           _quantile(edge_punct_counts, 0.90, 2.0),
        "unique_token_ratio_floor": _quantile(unique_token_ratios, 0.10, 0.0),
        "digit_ratio_cap":          _quantile(digit_ratios, 0.90, 1.0),
        "mean_token_len_floor":     _quantile(mean_token_lens, 0.10, 0.0),
        "support_floor":            _quantile(supports.astype(float).tolist(), 0.15, 0.0),
    }


# ── Pattern quality filter ───────────────────────────────────────────────

def passes_statistical_quality_filter(pattern, support, thresholds, role):
    """Check if a pattern is high-quality enough to become a signature.

    Returns True if it passes all checks. The thresholds dict comes from
    compute_statistical_quality_thresholds() above.
    """
    stats = pattern_shape_stats(pattern)
    token_count = int(stats["token_count"])

    # Must be at least a bigram
    if token_count < 2:
        return False

    # All-stopword patterns are never useful
    if stats["stopword_ratio"] >= 1.0:
        return False

    # Basic shape checks — reject patterns that are too punctuated, too
    # digit-heavy, have boundary punctuation, or have tiny tokens
    if stats["alpha_ratio"] < thresholds["alpha_ratio_floor"]:
        return False
    if stats["punct_ratio"] > thresholds["punct_ratio_cap"]:
        return False
    if stats["digit_ratio"] > thresholds["digit_ratio_cap"]:
        return False
    if stats["edge_punct_count"] > thresholds["edge_punct_cap"]:
        return False
    if stats["mean_token_len"] < thresholds["mean_token_len_floor"]:
        return False

    # Role-specific checks
    if role == "safe":
        # Safe sigs shouldn't have too many capitalised tokens (likely proper nouns)
        if stats["capitalized_token_ratio"] > thresholds["capitalized_ratio_cap"]:
            return False
        # Low-support + high-stopword = probably coincidental
        if support < thresholds["support_floor"] and stats["stopword_ratio"] > thresholds["stopword_ratio_cap"]:
            return False
    else:
        # Malicious: reject if high stopword AND low uniqueness (repetitive filler)
        if stats["stopword_ratio"] > thresholds["stopword_ratio_cap"] and stats["unique_token_ratio"] < thresholds["unique_token_ratio_floor"]:
            return False

    return True


# ── Generic pattern detection ────────────────────────────────────────────

def is_generic_pattern(pattern):
    """Return True if the pattern is too generic to be a useful malicious signature.

    Catches common English filler, char-boundary noise, and patterns that
    are just punctuation fragments.
    """
    stripped = pattern.strip()
    if not stripped:
        return True

    tokens = stripped.split()
    stats = pattern_shape_stats(stripped)

    # Single-token: needs to be mostly alphabetic and long enough
    if len(tokens) < 2:
        if stats["alpha_ratio"] < 0.6:
            return True
        if sum(c.isalpha() for c in stripped) < 8:
            return True
        return False

    # Pure stopword n-grams (e.g. "the and or")
    if all(t in STOPWORDS for t in tokens):
        return True

    # Char n-gram boundary noise: mostly punctuation or starts/ends with punct
    # e.g. ". to mak" or ", the fo" — these are artifacts from char vectorizer
    if stats["alpha_ratio"] < 0.55:
        return True
    if stats["edge_punct_count"] >= 2:
        return True

    return False
