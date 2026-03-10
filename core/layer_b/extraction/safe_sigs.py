"""SAFE allow-signature extraction for Layer B."""
import logging
import time

import numpy as np

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

    candidates = []
    for i in passing_indices:
        pattern = str(w_vocab[i])
        safe_df = int(w_safe_df[i])
        mal_df = int(w_mal_df[i])

        precision = safe_df / (safe_df + mal_df) if (safe_df + mal_df) else 0.0
        if precision < precision_threshold:
            continue

        candidates.append({"pattern": pattern, "safe_df": safe_df, "mal_df": mal_df, "precision": precision})

    # Sort by: precision (highest first), then support-per-token (favour concise
    # patterns), then raw support, then shortest pattern wins ties
    candidates.sort(
        key=lambda c: (
            c["precision"],
            c["safe_df"] / max(1, len(c["pattern"].split())),
            c["safe_df"],
            -len(c["pattern"]),
        ),
        reverse=True,
    )

    out = [
        {
            "pattern":           c["pattern"],
            "support":           c["safe_df"],
            "malicious_support": c["mal_df"],
            "safe_precision":    round(c["precision"], 4),
            "type":              "allow-safe",
        }
        for c in candidates
    ]
    log.info("SAFE signatures: %d selected (%.1fs)", len(out), time.perf_counter() - t0)
    return out
