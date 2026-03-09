"""SAFE allow-signature extraction for Layer B."""
import logging
import time

import numpy as np

from models.PatternStats import PatternStats
from core.layer_b.extraction.vectorise import (
    compute_statistical_quality_thresholds,
    passes_statistical_quality_filter,
)

log = logging.getLogger(__name__)


def build_safe_signatures(thresholds, w_vocab, w_safe_df, w_mal_df):
    """Build a list of high-precision safe allow-signature dicts.

    Only uses word n-grams (no char n-grams needed for safe sigs —
    safe patterns tend to be full phrases, not fragments).
    """
    t0 = time.perf_counter()

    safe_min_support    = thresholds["safe_min_support"]
    safe_mal_df_cap     = thresholds["safe_mal_df_cap"]
    precision_threshold = thresholds["safe_precision_threshold"]

    # Fast pre-filter with numpy
    mask = (w_safe_df >= safe_min_support) & (w_mal_df <= safe_mal_df_cap)
    passing_indices = np.nonzero(mask)[0]
    log.info("  %d / %d features pass support & DF-cap pre-filter", len(passing_indices), len(w_vocab))

    #Statistical quality filter 
    candidate_patterns = [str(w_vocab[i]) for i in passing_indices]
    candidate_supports = np.asarray([int(w_safe_df[i]) for i in passing_indices], dtype=np.int32)
    quality_thresholds = compute_statistical_quality_thresholds(
        candidate_patterns, candidate_supports, role="safe",
    )
    log.info("  SAFE quality thresholds: %s", quality_thresholds)

    candidates = []
    filtered_count = 0
    for i in passing_indices:
        pattern = str(w_vocab[i])
        safe_df = int(w_safe_df[i])
        mal_df = int(w_mal_df[i])

        if not passes_statistical_quality_filter(pattern, safe_df, quality_thresholds, role="safe"):
            filtered_count += 1
            continue

        precision = safe_df / (safe_df + mal_df) if (safe_df + mal_df) else 0.0
        if precision < precision_threshold:
            continue

        candidates.append(PatternStats(pattern, safe_df, mal_df))

    log.info("  SAFE statistical filter removed %d candidates", filtered_count)

    # Sort by: precision (highest first), then support-per-token (favour concise
    # patterns), then raw support, then shortest pattern wins ties
    candidates.sort(
        key=lambda c: (
            c.safe_precision,
            c.safe_df / max(1, len(c.pattern.split())),
            c.safe_df,
            -len(c.pattern),
        ),
        reverse=True,
    )

    # Build output dicts
    out = [
        {
            "pattern":           c.pattern,
            "support":           c.safe_df,
            "malicious_support": c.mal_df,
            "safe_precision":    round(c.safe_precision, 4),
            "type":              "allow-safe",
        }
        for c in candidates
    ]
    log.info("SAFE signatures: %d selected (%.1fs)", len(out), time.perf_counter() - t0)
    return out
