"""MALICIOUS block-signature extraction for Layer B."""
import gc
import logging
import time
import numpy as np

from models.PatternStats import PatternStats
from core.layer_b.extraction.vectorise import (
    compute_statistical_quality_thresholds,
    doc_freq,
    is_generic_pattern,
    make_vectorizers,
    passes_statistical_quality_filter,
)

log = logging.getLogger(__name__)


def build_malicious_signatures(texts_norm, labels, thresholds, w_vocab, w_safe_df, w_mal_df):
    """Build a list of high-precision malicious signature dicts.

    Uses both word n-grams (passed in from the caller) and char n-grams
    (fitted here) to find patterns that appear almost exclusively in
    malicious text.
    """
    t0 = time.perf_counter()

    # Unpack the thresholds we need
    min_df              = thresholds["vec_min_df"]
    max_features        = thresholds["vec_max_features"]
    mal_min_support     = thresholds["mal_min_support"]
    mal_safe_df_cap     = thresholds["mal_safe_df_cap"]
    precision_threshold = thresholds["mal_precision_threshold"]
    prevalence_cap_frac = thresholds["mal_prevalence_cap_frac"]
    min_pattern_len     = thresholds["mal_min_pattern_len"]

    y = labels.astype(int)
    safe_mask = (y == 0)
    mal_mask = (y == 1)

    log.info("Reusing word DF arrays (vocab=%d) for MALICIOUS signatures", len(w_vocab))

    # ── Fit a char n-gram vectorizer ─────────────────────────────────
    # Word n-grams catch multi-word phrases; char n-grams catch substrings
    # and partial matches that word boundaries would miss.
    _, char_vec = make_vectorizers(min_df=min_df, max_features=max_features)
    log.info("Fitting char vectoriser for MALICIOUS signatures …")
    char_matrix = char_vec.fit_transform(texts_norm)  # type: ignore[assignment]
    char_vocab = char_vec.get_feature_names_out()
    log.info("  char vocab=%d, nnz=%d", len(char_vocab), char_matrix.nnz)  # type: ignore[union-attr]

    char_safe_df = doc_freq(char_matrix[safe_mask])  # type: ignore[index]
    char_mal_df = doc_freq(char_matrix[mal_mask])  # type: ignore[index]
    del char_matrix
    gc.collect()

    # ── Merge word + char features into one big array ────────────────
    all_patterns = np.concatenate([w_vocab, char_vocab])
    all_safe_df = np.concatenate([w_safe_df, char_safe_df])
    all_mal_df = np.concatenate([w_mal_df, char_mal_df])
    del char_vocab, char_safe_df, char_mal_df
    gc.collect()

    # ── Fast vectorised pre-filter ───────────────────────────────────
    # This uses numpy boolean indexing to quickly eliminate the vast majority
    # of features before the slower Python-loop quality checks.
    total_df = all_safe_df + all_mal_df
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(total_df > 0, all_mal_df / total_df, 0.0)

    pattern_lengths = np.vectorize(len)(all_patterns)

    # If a pattern appears in >12% of malicious docs, it's generic English
    mal_prevalence_cap = int(prevalence_cap_frac * mal_mask.sum())

    mask = (
        (all_mal_df >= mal_min_support)
        & (all_mal_df <= mal_prevalence_cap)
        & (precision >= precision_threshold)
        & (all_safe_df <= mal_safe_df_cap)
        & (pattern_lengths >= min_pattern_len)
    )
    passing_indices = np.nonzero(mask)[0]
    log.info("  %d / %d features pass vectorised pre-filter", len(passing_indices), len(all_patterns))

    # ── Statistical quality filter (Python loop) ─────────────────────
    # For the survivors, compute pattern-shape stats and filter out generic
    # English, boundary noise, and other junk.
    candidate_patterns = [str(all_patterns[i]) for i in passing_indices]
    candidate_supports = np.asarray([int(all_mal_df[i]) for i in passing_indices], dtype=np.int32)
    quality_thresholds = compute_statistical_quality_thresholds(
        candidate_patterns, candidate_supports, role="malicious",
    )
    log.info("  MALICIOUS quality thresholds: %s", quality_thresholds)

    selected = []
    filtered_count = 0
    for i in passing_indices:
        pattern = str(all_patterns[i])

        if is_generic_pattern(pattern):
            filtered_count += 1
            continue
        if not passes_statistical_quality_filter(pattern, int(all_mal_df[i]), quality_thresholds, role="malicious"):
            filtered_count += 1
            continue

        selected.append(PatternStats(
            pattern=pattern,
            safe_df=int(all_safe_df[i]),
            mal_df=int(all_mal_df[i]),
        ))

    log.info("  MALICIOUS statistical filter removed %d candidates", filtered_count)

    # Sort: highest precision first, then highest support, then shortest pattern
    selected.sort(key=lambda s: (s.mal_precision, s.mal_df, -len(s.pattern)), reverse=True)

    # Build output dicts
    out = [
        {
            "pattern":      s.pattern,
            "precision":    round(s.mal_precision, 4),
            "support":      s.mal_df,
            "safe_support": s.safe_df,
            "type":         "block-high",
        }
        for s in selected
    ]
    log.info("MALICIOUS signatures: %d selected (%.1fs)", len(out), time.perf_counter() - t0)
    return out
