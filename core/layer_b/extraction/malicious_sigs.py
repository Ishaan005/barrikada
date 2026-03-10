"""MALICIOUS block-signature extraction for Layer B."""
import gc
import logging
import time
import numpy as np

from core.layer_b.extraction.vectorise import (
    doc_freq,
    make_vectorizers,
)

log = logging.getLogger(__name__)


def build_malicious_signatures(
    texts_norm, labels, thresholds, w_vocab, w_safe_df, w_mal_df,
):
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

    selected = []
    for i in passing_indices:
        pattern = str(all_patterns[i])
        safe_df = int(all_safe_df[i])
        mal_df = int(all_mal_df[i])

        total_df = safe_df + mal_df
        selected.append({
            "pattern":      pattern,
            "precision":    round(mal_df / total_df if total_df else 0.0, 4),
            "support":      mal_df,
            "safe_support": safe_df,
            "type":         "block-high",
        })

    # Sort: highest precision first, then highest support, then shortest pattern
    selected.sort(key=lambda d: (d["precision"], d["support"], -len(d["pattern"])), reverse=True)

    log.info("MALICIOUS signatures: %d selected (%.1fs)", len(selected), time.perf_counter() - t0)
    return selected
